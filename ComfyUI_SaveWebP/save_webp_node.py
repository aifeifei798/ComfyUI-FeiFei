import os
import torch
import numpy as np
from PIL import Image
from datetime import datetime
import folder_paths

class SaveWebPWithTimestamp:
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
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
    CATEGORY = "image"

    def save_images(self, images, quality, lossless, subdir):
        # 确定保存路径
        full_output_folder = os.path.join(self.output_dir, subdir)
        if not os.path.exists(full_output_folder):
            os.makedirs(full_output_folder)

        results = list()
        
        for image in images:
            # 将张量转换为 PIL Image
            i = 255. * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            
            # 生成文件名: YYYYMMDD_HHMMSS_毫秒.webp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            file_name = f"{timestamp}.webp"
            file_path = os.path.join(full_output_folder, file_name)
            
            # 保存为 WebP
            img.save(file_path, format="WEBP", quality=quality, lossless=lossless)
            
            results.append({
                "filename": file_name,
                "subfolder": subdir,
                "type": self.type
            })

        return {"ui": {"images": results}}

NODE_CLASS_MAPPINGS = {
    "SaveWebPWithTimestamp": SaveWebPWithTimestamp
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SaveWebPWithTimestamp": "Save WebP (Timestamp)"
}
