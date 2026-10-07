# -----------------------------------------------------------------
# A ComfyUI custom node
# Adds a three-line watermark to an image, with a configurable font size per line.
# -----------------------------------------------------------------

import os
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont


# Cross-platform font lookup chain: the font_path input first, then common system fonts
DEFAULT_FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "C:/Windows/Fonts/msyh.ttc",
    "C:/Windows/Fonts/simhei.ttf",
]


def _resolve_font_path(font_path):
    """Return a usable font path; falls back through the candidate chain when the
    user path is invalid, and returns None when every candidate fails"""
    candidates = []
    if isinstance(font_path, str) and font_path.strip():
        candidates.append(font_path.strip())
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
        print(f"Warning: failed to load font {resolved_path} ({e}), {line_label} falls back to default font.")
    if original_path and original_path != resolved_path:
        print(f"Warning: font file not found {original_path}, {line_label} uses {resolved_path or 'PIL default font'}.")
    return ImageFont.load_default()


# -----------------------------------------------------------------
# Core watermark function (now three lines)
# -----------------------------------------------------------------
def add_watermark(
    image,
    font_path,
    watermark_text_line1="FeiFei",
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

    # Avoid mutating the caller's RGBA object in place
    if image.mode != "RGBA":
        image = image.convert("RGBA")
    else:
        image = image.copy()

    width, height = image.size
    draw = ImageDraw.Draw(image)

    resolved = _resolve_font_path(font_path)

    # --- 1. Load the three fonts ---
    font1 = _load_font(resolved, font_size_line1, "Line 1", font_path)
    font2 = _load_font(resolved, font_size_line2, "Line 2", font_path)
    font3 = _load_font(resolved, font_size_line3, "Line 3", font_path)

    # --- 2. Measure each line ---
    bbox1 = draw.textbbox((0, 0), watermark_text_line1, font=font1)
    text_width_line1 = bbox1[2] - bbox1[0]
    text_height_line1 = bbox1[3] - bbox1[1]

    bbox2 = draw.textbbox((0, 0), watermark_text_line2, font=font2)
    text_width_line2 = bbox2[2] - bbox2[0]
    text_height_line2 = bbox2[3] - bbox2[1]

    bbox3 = draw.textbbox((0, 0), watermark_text_line3, font=font3)
    text_width_line3 = bbox3[2] - bbox3[0]
    text_height_line3 = bbox3[3] - bbox3[1]

    # --- 3. Position the three lines (bottom-right aligned, clamped on screen) ---
    margin = 20
    line_spacing = 10

    def _clamp_x(text_width):
        # Text wider than the image is left-aligned to margin, so no negative coordinate clips it
        if text_width + margin * 2 >= width:
            return margin
        return max(margin, width - text_width - margin)

    def _clamp_y(y):
        return max(margin, y)

    # Third line sits lowest
    y_line3 = _clamp_y(height - text_height_line3 - margin)
    x_line3 = _clamp_x(text_width_line3)

    # Second line above the third
    y_line2 = _clamp_y(y_line3 - text_height_line2 - line_spacing)
    x_line2 = _clamp_x(text_width_line2)

    # First line above the second
    y_line1 = _clamp_y(y_line2 - text_height_line1 - line_spacing)
    x_line1 = _clamp_x(text_width_line1)

    # Opaque pure white plus a black stroke keeps the text readable on bright backgrounds
    white_color = (255, 255, 255, 255)
    stroke_fill = (0, 0, 0, 200)

    # --- 4. Draw the text ---
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
# ComfyUI node class
# -----------------------------------------------------------------
class WatermarkNode:
    @classmethod
    def INPUT_TYPES(cls):
        """Declare the node's inputs (now including the third line)"""
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
        """Core execution logic"""

        if not isinstance(image, torch.Tensor):
            raise ValueError(f"image must be a ComfyUI IMAGE Tensor [B,H,W,C], got {type(image)}")
        if image.ndim == 3:
            image = image.unsqueeze(0)
        if image.ndim != 4:
            raise ValueError(f"Bad image dims, expected [B,H,W,C], got {tuple(image.shape)}")

        watermarked_images = []
        for i in range(image.shape[0]):
            # 1. Pull one tensor out of the batch
            img_tensor = image[i]

            # 2. Convert to a NumPy array and rescale 0-1 -> 0-255
            img_np = np.clip(255.0 * img_tensor.cpu().numpy(), 0, 255).astype(np.uint8)
            # Tolerate grayscale/single channel: (H,W,1) -> (H,W); (H,W,2) zero-padded to 3 channels
            if img_np.ndim == 3 and img_np.shape[-1] == 1:
                img_np = img_np[..., 0]
            elif img_np.ndim == 3 and img_np.shape[-1] == 2:
                pad = np.zeros_like(img_np[..., :1])
                img_np = np.concatenate([img_np, pad], axis=-1)

            # 3. Build a PIL image from the NumPy array
            pil_image = Image.fromarray(img_np)

            # 4. Call the core function to add the three watermark lines
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

            # 5. Convert the processed PIL image back to a NumPy array
            img_np_watermarked = np.array(pil_image_watermarked).astype(np.float32)

            # 6. Rescale back to 0-1 and convert back to a PyTorch tensor
            img_tensor_watermarked = torch.from_numpy(img_np_watermarked / 255.0)

            watermarked_images.append(img_tensor_watermarked)

        # Stack the processed images back into one batch tensor
        final_tensor = torch.stack(watermarked_images)

        return (final_tensor,)


# -----------------------------------------------------------------
# Mapping dicts ComfyUI requires
# -----------------------------------------------------------------
NODE_CLASS_MAPPINGS = {"WatermarkNode": WatermarkNode}

NODE_DISPLAY_NAME_MAPPINGS = {"WatermarkNode": "Watermark"}
