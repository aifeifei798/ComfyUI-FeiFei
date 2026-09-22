# -----------------------------------------------------------------
# 这是一个ComfyUI的自定义节点
# 功能：为图像添加一个三行、可自定义字体大小的水印。
# -----------------------------------------------------------------

import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont


# -----------------------------------------------------------------
# 核心水印添加函数 (已升级为 3 行)
# -----------------------------------------------------------------
def add_watermark(
    image,
    font_path,
    watermark_text_line1="妃妃",
    font_size_line1=20,
    watermark_text_line2="@aifeifei799",
    font_size_line2=20,
    watermark_text_line3="AI Generated",
    font_size_line3=20,
):
    """
    Adds a three-line watermark to an image with individually adjustable font sizes.
    """
    if not isinstance(image, Image.Image):
        raise ValueError("Input must be a PIL Image object")

    if image.mode != "RGBA":
        image = image.convert("RGBA")

    width, height = image.size
    draw = ImageDraw.Draw(image)

    # --- 1. 加载三个字体 ---
    try:
        font1 = ImageFont.truetype(font_path, font_size_line1)
    except IOError:
        print(f"警告：找不到字体文件 {font_path}。第一行将使用默认字体。")
        font1 = ImageFont.load_default()

    try:
        font2 = ImageFont.truetype(font_path, font_size_line2)
    except IOError:
        print(f"警告：找不到字体文件 {font_path}。第二行将使用默认字体。")
        font2 = ImageFont.load_default()

    try:
        font3 = ImageFont.truetype(font_path, font_size_line3)
    except IOError:
        print(f"警告：找不到字体文件 {font_path}。第三行将使用默认字体。")
        font3 = ImageFont.load_default()

    # --- 2. 计算每一行的文字尺寸 ---
    bbox1 = draw.textbbox((0, 0), watermark_text_line1, font=font1)
    text_width_line1 = bbox1[2] - bbox1[0]
    text_height_line1 = bbox1[3] - bbox1[1]

    bbox2 = draw.textbbox((0, 0), watermark_text_line2, font=font2)
    text_width_line2 = bbox2[2] - bbox2[0]
    text_height_line2 = bbox2[3] - bbox2[1]

    bbox3 = draw.textbbox((0, 0), watermark_text_line3, font=font3)
    text_width_line3 = bbox3[2] - bbox3[0]
    text_height_line3 = bbox3[3] - bbox3[1]

    # --- 3. 计算三行文字的位置 (右下角对齐) ---
    margin = 20
    line_spacing = 10

    # 第三行在最下方
    y_line3 = height - text_height_line3 - margin
    x_line3 = width - text_width_line3 - margin

    # 第二行在第三行上方
    y_line2 = y_line3 - text_height_line2 - line_spacing
    x_line2 = width - text_width_line2 - margin

    # 第一行在第二行上方
    y_line1 = y_line2 - text_height_line1 - line_spacing
    x_line1 = width - text_width_line1 - margin

    # 纯白色且不透明
    white_color = (255, 255, 255, 255)

    # --- 4. 绘制文字 ---
    draw.text((x_line1, y_line1), watermark_text_line1, font=font1, fill=white_color)
    draw.text((x_line2, y_line2), watermark_text_line2, font=font2, fill=white_color)
    draw.text((x_line3, y_line3), watermark_text_line3, font=font3, fill=white_color)

    return image.convert("RGB")


# -----------------------------------------------------------------
# ComfyUI 节点类
# -----------------------------------------------------------------
class WatermarkNode:
    @classmethod
    def INPUT_TYPES(cls):
        """定义节点的输入 (已增加第三行输入配置)"""
        return {
            "required": {
                "image": ("IMAGE",),
                "font_path": (
                    "STRING",
                    {
                        "multiline": False,
                        "default": "D:\\ComfyUI_windows_portable\\ComfyUI\\Fonts\\Iansui-Regular.ttf",
                    },
                ),
                "text_line1": ("STRING", {"multiline": False, "default": "Feimatrix"}),
                "font_size_line1": (
                    "INT",
                    {"default": 40, "min": 1, "max": 1024, "step": 1},
                ),
                "text_line2": (
                    "STRING",
                    {"multiline": False, "default": "Generated media"},
                ),
                "font_size_line2": (
                    "INT",
                    {"default": 30, "min": 1, "max": 1024, "step": 1},
                ),
                "text_line3": (
                    "STRING",
                    {"multiline": False, "default": "AI Generated"},
                ),
                "font_size_line3": (
                    "INT",
                    {"default": 20, "min": 1, "max": 1024, "step": 1},
                ),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply_watermark"
    CATEGORY = "FeiFei"

    def apply_watermark(
        self,
        image,
        font_path,
        text_line1,
        font_size_line1,
        text_line2,
        font_size_line2,
        text_line3,
        font_size_line3,
    ):
        """节点的核心执行逻辑"""

        watermarked_images = []
        for i in range(image.shape[0]):
            # 1. 从批次中取出一个张量
            img_tensor = image[i]

            # 2. 将张量转换为 NumPy 数组并调整数值范围 (0-1 -> 0-255)
            img_np = np.clip(255.0 * img_tensor.cpu().numpy(), 0, 255).astype(np.uint8)

            # 3. 从 NumPy 数组创建 PIL 图像
            pil_image = Image.fromarray(img_np)

            # 4. 调用核心函数添加三行水印
            pil_image_watermarked = add_watermark(
                pil_image,
                font_path,
                text_line1,
                font_size_line1,
                text_line2,
                font_size_line2,
                text_line3,
                font_size_line3,
            )

            # 5. 将处理后的 PIL 图像转换回 NumPy 数组
            img_np_watermarked = np.array(pil_image_watermarked).astype(np.float32)

            # 6. 将数值范围调回 0-1 并转换回 PyTorch 张量
            img_tensor_watermarked = torch.from_numpy(img_np_watermarked / 255.0)

            watermarked_images.append(img_tensor_watermarked)

        # 将处理后的图像列表堆叠成一个批次张量
        final_tensor = torch.stack(watermarked_images)

        return (final_tensor,)


# -----------------------------------------------------------------
# ComfyUI 必须的映射字典
# -----------------------------------------------------------------
NODE_CLASS_MAPPINGS = {"WatermarkNode": WatermarkNode}

NODE_DISPLAY_NAME_MAPPINGS = {"WatermarkNode": "图像水印 (Watermark)"}
