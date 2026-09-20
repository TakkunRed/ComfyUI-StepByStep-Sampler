import glob
import os
import uuid

import numpy as np
from PIL import Image
import folder_paths

_SUBFOLDER = "step_by_step"


def save_step_images(images, node_id, kind):
    """
    画像テンソル [B,H,W,C] を temp フォルダに JPEG で保存し、フロントエンド用の参照リストを返す。

    base64 を ui / WebSocket / 履歴に載せると、ステップ数ぶんの画像が毎回サーバーの履歴に残り
    メッセージも巨大になるため、PreviewImage と同様にファイル参照だけを渡す。
    同じノードの前回分は削除するので、実行を重ねてもファイルは増えない。
    """
    out_dir = os.path.join(folder_paths.get_temp_directory(), _SUBFOLDER)
    os.makedirs(out_dir, exist_ok=True)
    node_id = "".join(c for c in str(node_id) if c.isalnum() or c in "-_") or "0"
    prefix = f"{kind}_{node_id}_"

    for old in glob.glob(os.path.join(out_dir, prefix + "*")):
        try:
            os.remove(old)
        except OSError:
            pass

    run_id = uuid.uuid4().hex[:8]
    refs = []
    for i, img in enumerate(images):
        arr = np.clip(img.cpu().numpy() * 255.0, 0, 255).astype(np.uint8)
        if arr.ndim == 3 and arr.shape[-1] == 1:
            arr = arr[:, :, 0]
        elif arr.ndim == 3 and arr.shape[-1] > 3:
            arr = arr[:, :, :3]  # アルファは JPEG に保存できない
        filename = f"{prefix}{run_id}_{i:04d}.jpg"
        Image.fromarray(arr).save(os.path.join(out_dir, filename), format="JPEG", quality=90)
        refs.append({"filename": filename, "subfolder": _SUBFOLDER, "type": "temp"})
    return refs


# 1. 再生用ビューアノード
class StepStepPlayer:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "images": ("IMAGE",),
                },
                "hidden": {"unique_id": "UNIQUE_ID"}}

    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "view_images"
    CATEGORY = "custom_nodes/viewers"

    def view_images(self, images, unique_id=None):
        return {"ui": {"step_images": save_step_images(images, unique_id, "player")}}

# 2. 比較用ビューアノード
class StepStepComparer:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "images": ("IMAGE",),
                },
                "hidden": {"unique_id": "UNIQUE_ID"}}

    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "view_images"
    CATEGORY = "custom_nodes/viewers"

    def view_images(self, images, unique_id=None):
        return {"ui": {"step_images": save_step_images(images, unique_id, "comparer")}}

# --- ノードの登録設定 ---

NODE_CLASS_MAPPINGS = {
    "StepStepPlayer": StepStepPlayer,
    "StepStepComparer": StepStepComparer
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "StepStepPlayer": "Step-by-Step Player",
    "StepStepComparer": "Step-by-Step Comparer"
}
