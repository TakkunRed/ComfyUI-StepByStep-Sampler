import torch
import numpy as np
from PIL import Image
import io
import base64
from server import PromptServer

def encode_images_to_base64(images):
    """画像テンソルのバッチをBase64文字列のリストに変換する共通関数"""
    encoded_images = []
    for img in images:
        # Tensor (B, H, W, C) -> Numpy -> PIL
        i = 255. * img.cpu().numpy()
        pil_img = Image.fromarray(np.uint8(i))
        
        buffered = io.BytesIO()
        pil_img.save(buffered, format="JPEG", quality=85)
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        encoded_images.append(f"data:image/jpeg;base64,{img_str}")
    return encoded_images

# 1. 再生用ビューアノード
class StepStepPlayer:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "images": ("IMAGE",), 
                }}

    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "view_images"
    CATEGORY = "custom_nodes/viewers"

    def view_images(self, images):
        encoded = encode_images_to_base64(images)
        # フロントエンドの Player 側に送信
        PromptServer.instance.send_json("step_player_update", {"images": encoded})
        return {"ui": {"images": encoded}}

# 2. 比較用ビューアノード
class StepStepComparer:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "images": ("IMAGE",), 
                }}

    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "view_images"
    CATEGORY = "custom_nodes/viewers"

    def view_images(self, images):
        encoded = encode_images_to_base64(images)
        # フロントエンドの Comparer 側に送信
        PromptServer.instance.send_json("step_comparer_update", {"images": encoded})
        return {"ui": {"images": encoded}}

# --- ノードの登録設定 ---

NODE_CLASS_MAPPINGS = {
    "StepStepPlayer": StepStepPlayer,
    "StepStepComparer": StepStepComparer
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "StepStepPlayer": "Step-by-Step Player",
    "StepStepComparer": "Step-by-Step Comparer"
}