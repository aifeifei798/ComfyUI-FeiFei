import os
import re
import torch
import numpy as np
from PIL import Image
from datetime import datetime

DEFAULT_SUBDIR = "webp_outputs"


def _get_output_directory():
    """懒加载 folder_paths， standalone 测试时降级到 ./output"""
    try:
        import folder_paths
        return folder_paths.get_output_directory()
    except Exception:
        fallback = os.path.join(os.getcwd(), "output")
        print(f"[SaveWebP] folder_paths 不可用，降级到 {fallback}")
        return fallback


def _sanitize_subdir(subdir):
    """防止 ../ 目录穿越与绝对路径，非法输入回退默认值"""
    if not isinstance(subdir, str) or not subdir.strip():
        return DEFAULT_SUBDIR
    cleaned = subdir.strip().replace("\\", "/")
    if os.path.isabs(cleaned):
        return DEFAULT_SUBDIR
    parts = [p for p in cleaned.split("/") if p not in ("", ".", "..")]
    # 仅允许 安全字符
    safe_parts = [p for p in parts if re.fullmatch(r"[\w\-. ]+", p)]
    if not safe_parts:
        return DEFAULT_SUBDIR
    return os.path.join(*safe_parts)

class SaveWebPWithTimestamp:
    def __init__(self):
        self.output_dir = _get_output_directory()
        self.type = "output"
        self.prefix_append = ""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "quality": ("INT", {"default": 90, "min": 1, "max": 100, "step": 1}),
                "lossless": ("BOOLEAN", {"default": False}),
                "subdir": ("STRING", {"default": "webp_outputs"}),
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "save_images"
    OUTPUT_NODE = True
    CATEGORY = "FeiFei"

    def save_images(self, images, quality, lossless, subdir):
        # 确定保存路径（防穿越）
        safe_subdir = _sanitize_subdir(subdir)
        full_output_folder = os.path.join(self.output_dir, safe_subdir)
        os.makedirs(full_output_folder, exist_ok=True)

        results = list()
        # 同一批次用同一时间戳前缀 + 序号，保证毫秒内多图不覆盖
        batch_stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]

        for idx, image in enumerate(images):
            # 将张量转换为 PIL Image
            i = 255. * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            if img.mode not in ("RGB", "RGBA"):
                img = img.convert("RGB")

            # 生成文件名: YYYYMMDD_HHMMSS_毫秒_序号.webp
            file_name = f"{batch_stamp}_{idx:03d}.webp"
            file_path = os.path.join(full_output_folder, file_name)

            # 保存为 WebP（lossless 时不传 quality，避免 Pillow 告警/忽略）
            if lossless:
                img.save(file_path, format="WEBP", lossless=True)
            else:
                img.save(file_path, format="WEBP", quality=int(quality), lossless=False)

            results.append({
                "filename": file_name,
                "subfolder": safe_subdir,
                "type": self.type
            })

        return {"ui": {"images": results}}

NODE_CLASS_MAPPINGS = {
    "SaveWebPWithTimestamp": SaveWebPWithTimestamp
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SaveWebPWithTimestamp": "Save WebP (Timestamp)"
}
