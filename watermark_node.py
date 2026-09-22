# -----------------------------------------------------------------
# 这是一个ComfyUI的自定义节点
# 功能：为图像添加一个三行、可自定义字体大小的水印。
# -----------------------------------------------------------------

import os
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont


# 跨平台字体查找链：用户路径优先，其次常见系统字体
DEFAULT_FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "C:/Windows/Fonts/msyh.ttc",
    "C:/Windows/Fonts/simhei.ttf",
    "D:/ComfyUI_windows_portable/ComfyUI/Fonts/Iansui-Regular.ttf",
]


def _resolve_font_path(font_path):
    """返回可用字体路径；用户路径无效时按候选链回退，全部失效返回 None"""
    candidates = []
    if isinstance(font_path, str) and font_path.strip():
        candidates.append(font_path.strip())
    env_font = os.environ.get("FEIFEI_FONT_PATH", "").strip()
    if env_font:
        candidates.append(env_font)
    candidates.extend(DEFAULT_FONT_CANDIDATES)
    for cand in candidates:
        if cand and os.path.isfile(cand):
            return cand
    return None


def _load_font(resolved_path, size, line_label, original_path):
    try:
        if resolved_path:
            return ImageFont.truetype(resolved_path, size)
    except (IOError, OSError) as e:
        print(f"警告：字体 {resolved_path} 加载失败 ({e})，{line_label}回退默认字体。")
    if original_path and original_path != resolved_path:
        print(f"警告：找不到字体文件 {original_path}，{line_label}使用 {resolved_path or 'PIL默认字体'}。")
    return ImageFont.load_default()


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

    # 避免原地修改调用方的 RGBA 对象
    if image.mode != "RGBA":
        image = image.convert("RGBA")
    else:
        image = image.copy()

    width, height = image.size
    draw = ImageDraw.Draw(image)

    resolved = _resolve_font_path(font_path)

    # --- 1. 加载三个字体 ---
    font1 = _load_font(resolved, font_size_line1, "第一行将", font_path)
    font2 = _load_font(resolved, font_size_line2, "第二行将", font_path)
    font3 = _load_font(resolved, font_size_line3, "第三行将", font_path)

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

    # --- 3. 计算三行文字的位置 (右下角对齐，钳制防出屏) ---
    margin = 20
    line_spacing = 10

    def _clamp_x(text_width):
        # 文字比图宽时左对齐到 margin，避免负坐标裁掉
        if text_width + margin * 2 >= width:
            return margin
        return max(margin, width - text_width - margin)

    def _clamp_y(y):
        return max(margin, y)

    # 第三行在最下方
    y_line3 = _clamp_y(height - text_height_line3 - margin)
    x_line3 = _clamp_x(text_width_line3)

    # 第二行在第三行上方
    y_line2 = _clamp_y(y_line3 - text_height_line2 - line_spacing)
    x_line2 = _clamp_x(text_width_line2)

    # 第一行在第二行上方
    y_line1 = _clamp_y(y_line2 - text_height_line1 - line_spacing)
    x_line1 = _clamp_x(text_width_line1)

    # 纯白色且不透明 + 黑色描边保证亮底可读
    white_color = (255, 255, 255, 255)
    stroke_fill = (0, 0, 0, 200)

    # --- 4. 绘制文字 ---
    for (x, y, txt, font, size) in (
        (x_line1, y_line1, watermark_text_line1, font1, font_size_line1),
        (x_line2, y_line2, watermark_text_line2, font2, font_size_line2),
        (x_line3, y_line3, watermark_text_line3, font3, font_size_line3),
    ):
        if not txt:
            continue
        stroke_w = max(1, int(size // 20))
        draw.text((x, y), txt, font=font, fill=white_color,
                  stroke_width=stroke_w, stroke_fill=stroke_fill)

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
                        "default": "",
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

        if not isinstance(image, torch.Tensor):
            raise ValueError("image 必须是 ComfyUI IMAGE Tensor [B,H,W,C]")
        if image.ndim == 3:
            image = image.unsqueeze(0)
        if image.ndim != 4:
            raise ValueError(f"image 维度异常，期望 [B,H,W,C]，实际 {tuple(image.shape)}")

        watermarked_images = []
        for i in range(image.shape[0]):
            # 1. 从批次中取出一个张量
            img_tensor = image[i]

            # 2. 将张量转换为 NumPy 数组并调整数值范围 (0-1 -> 0-255)
            img_np = np.clip(255.0 * img_tensor.cpu().numpy(), 0, 255).astype(np.uint8)
            # 兼容灰度/单通道：(H,W,1) -> (H,W)；(H,W,2) 补零成 3 通道
            if img_np.ndim == 3 and img_np.shape[-1] == 1:
                img_np = img_np[..., 0]
            elif img_np.ndim == 3 and img_np.shape[-1] == 2:
                pad = np.zeros_like(img_np[..., :1])
                img_np = np.concatenate([img_np, pad], axis=-1)

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
