import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import nodes
import comfy.samplers
import comfy.sample
import comfy.utils
import comfy.model_management
import latent_preview
from server import PromptServer

DIFF_METHODS = ["MSE", "RMSE", "L1", "PSNR", "SSIM"]


class StepByStepSampler:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "model":         ("MODEL",),
            "seed":          ("INT",   {"default": 0,   "min": 0,   "max": 0xffffffffffffffff}),
            "steps":         ("INT",   {"default": 20,  "min": 1,   "max": 10000}),
            "cfg":           ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step": 0.1}),
            "sampler_name":  (comfy.samplers.KSampler.SAMPLERS,),
            "scheduler":     (comfy.samplers.KSampler.SCHEDULERS,),
            "positive":      ("CONDITIONING",),
            "negative":      ("CONDITIONING",),
            "latent_image":  ("LATENT",),
            "denoise":       ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            "vae":           ("VAE",),
            "save_interval": ("INT",   {"default": 1,   "min": 1,   "max": 100}),
            "show_overlay":  ("BOOLEAN", {"default": True, "label_on": "ON", "label_off": "OFF"}),
            "diff_method":   (DIFF_METHODS, {"default": "MSE"}),
        }}

    RETURN_TYPES  = ("LATENT", "IMAGE",       "IMAGE")
    RETURN_NAMES  = ("LATENT", "STEP_IMAGES", "LAST_IMAGE")
    FUNCTION = "sample"
    CATEGORY = "custom_nodes/sampling"

    # ------------------------------------------------------------------ #
    #  x0 スケール変換
    # ------------------------------------------------------------------ #
    def _process_x0(self, model, x0):
        try:
            return model.model.process_latent_out(x0.to(torch.float32))
        except Exception:
            return x0.to(torch.float32)

    # ------------------------------------------------------------------ #
    #  VAE デコード
    # ------------------------------------------------------------------ #
    def _safe_decode(self, vae, latent):
        decoded = vae.decode(latent.float())
        if decoded.dim() == 5:
            decoded = decoded[:, 0, :, :, :]
        return decoded  # [B, H, W, C]

    # ------------------------------------------------------------------ #
    #  差分計算（latent 空間・CPU）
    # ------------------------------------------------------------------ #
    def _calc_diff(self, a, b, method):
        af = a.float().reshape(-1)
        bf = b.float().reshape(-1)
        n  = min(len(af), len(bf))
        af, bf = af[:n], bf[:n]

        if method == "L1":
            return (af - bf).abs().mean().item()
        elif method == "MSE":
            return ((af - bf) ** 2).mean().item()
        elif method == "RMSE":
            return ((af - bf) ** 2).mean().sqrt().item()
        elif method == "PSNR":
            mse = ((af - bf) ** 2).mean().item()
            if mse < 1e-10:
                return float("inf")
            return 20 * np.log10(6.0) - 10 * np.log10(mse)
        elif method == "SSIM":
            mu_a  = af.mean()
            mu_b  = bf.mean()
            sig_a = af.var()
            sig_b = bf.var()
            sig_ab = ((af - mu_a) * (bf - mu_b)).mean()
            C1, C2 = 0.01 ** 2, 0.03 ** 2
            ssim = ((2 * mu_a * mu_b + C1) * (2 * sig_ab + C2)) / \
                   ((mu_a ** 2 + mu_b ** 2 + C1) * (sig_a + sig_b + C2))
            return ssim.item()
        return 0.0

    def _format_diff(self, value, method):
        if method == "PSNR":
            return "∞ dB" if value == float("inf") else f"{value:.2f} dB"
        return f"{value:.4f}"

    # ------------------------------------------------------------------ #
    #  テキストオーバーレイ
    # ------------------------------------------------------------------ #
    def _annotate(self, decoded, label):
        img_np = decoded[0].cpu().float().numpy()
        img_np = np.clip(img_np, 0.0, 1.0)
        img    = Image.fromarray(np.uint8(img_np * 255.0))
        draw   = ImageDraw.Draw(img)
        w, h   = img.size

        fontsize = max(16, int(h * 0.06))
        font = None
        for name in ("arialbd.ttf", "arial.ttf", "DejaVuSans-Bold.ttf", "DejaVuSans.ttf"):
            try:
                font = ImageFont.truetype(name, fontsize)
                break
            except (IOError, OSError):
                continue
        if font is None:
            font = ImageFont.load_default()

        pos = (int(w * 0.03), int(h * 0.03))
        for dx in range(-2, 3):
            for dy in range(-2, 3):
                if dx == 0 and dy == 0:
                    continue
                draw.text((pos[0]+dx, pos[1]+dy), label, fill=(0, 0, 0), font=font)
        draw.text(pos, label, fill=(255, 255, 255), font=font)

        out = np.array(img).astype(np.float32) / 255.0
        return torch.from_numpy(out).unsqueeze(0)  # [1, H, W, C]

    # ------------------------------------------------------------------ #
    #  メイン
    # ------------------------------------------------------------------ #
    def sample(self, model, seed, steps, cfg, sampler_name, scheduler,
               positive, negative, latent_image, denoise, vae, save_interval,
               show_overlay=True, diff_method="L1"):

        latent = latent_image["samples"]
        latent = comfy.sample.fix_empty_latent_channels(
            model, latent, latent_image.get("downscale_ratio_spacial", None))

        batch_inds = latent_image.get("batch_index", None)
        noise      = comfy.sample.prepare_noise(latent, seed, batch_inds)
        noise_mask = latent_image.get("noise_mask", None)

        preview_callback = latent_preview.prepare_callback(model, steps)
        self.step_images = []
        last_saved_step  = [-1]
        prev_x0_cpu      = [None]

        def callback(step, x0, x, total_steps):
            preview_callback(step, x0, x, total_steps)

            is_interval = (step % save_interval == 0)
            is_last     = (step == total_steps - 1)
            if not (is_interval or is_last):
                return
            if step == last_saved_step[0]:
                return
            last_saved_step[0] = step

            with torch.no_grad():
                # スケール変換
                x0_proc = self._process_x0(model, x0)
                x0_cpu  = x0_proc.cpu()

                # 差分計算（CPU）
                if prev_x0_cpu[0] is None:
                    diff_str = "-"
                else:
                    val      = self._calc_diff(x0_cpu, prev_x0_cpu[0], diff_method)
                    diff_str = self._format_diff(val, diff_method)
                prev_x0_cpu[0] = x0_cpu

                # VAE デコード
                try:
                    decoded = self._safe_decode(vae, x0_proc).cpu()
                except Exception as e:
                    print(f"[StepSampler] VAE decode failed at step {step}: {e}")
                    return

                # オーバーレイ
                label = f"Step {step + 1}/{total_steps}  {diff_method}: {diff_str}"
                img   = self._annotate(decoded, label) if show_overlay else decoded[0:1].cpu()
                self.step_images.append(img)

        disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED

        samples = comfy.sample.sample(
            model, noise, steps, cfg, sampler_name, scheduler,
            positive, negative, latent,
            denoise=denoise, noise_mask=noise_mask,
            callback=callback, disable_pbar=disable_pbar, seed=seed,
        )

        # ------------------------------------------------------------------ #
        #  出力整形
        # ------------------------------------------------------------------ #
        latent_out = latent_image.copy()
        latent_out["samples"] = samples

        # LAST_IMAGE：最終サンプル結果を VAE デコード（オーバーレイなし）
        try:
            last_image = self._safe_decode(vae, samples).cpu()
        except Exception as e:
            print(f"[StepSampler] final decode failed: {e}")
            last_image = torch.zeros(1, 64, 64, 3)

        if not self.step_images:
            # フォールバック：step_images が空の場合は last_image を流用
            label     = f"Step {steps}/{steps}  {diff_method}: -"
            step_imgs = self._annotate(last_image, label) if show_overlay else last_image[0:1]
            return (latent_out, step_imgs, last_image)

        return (latent_out, torch.cat(self.step_images, dim=0), last_image)

    # ------------------------------------------------------------------ #
    #  ユーティリティ（後方互換）
    # ------------------------------------------------------------------ #
    def process_and_annotate(self, image_tensor, text, auto_normalize=False):
        img_np = image_tensor[0].cpu().numpy()
        if auto_normalize:
            lo, hi = img_np.min(), img_np.max()
            img_np = (img_np - lo) / (hi - lo + 1e-5)
        return self._annotate(torch.from_numpy(img_np).unsqueeze(0), text)

    def preview_to_base64(self, image_tensor):
        import io, base64
        i   = 255.0 * image_tensor[0].cpu().numpy()
        img = Image.fromarray(np.uint8(i))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=70)
        return base64.b64encode(buf.getvalue()).decode("utf-8")


NODE_CLASS_MAPPINGS        = {"StepByStepSampler": StepByStepSampler}
NODE_DISPLAY_NAME_MAPPINGS = {"StepByStepSampler": "Step-by-Step Sampler"}
