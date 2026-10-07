"""Safe-area overlay node: framing check before the cinematic crop.

Burns a preview overlay onto the image: rule-of-thirds grid, action-safe
rectangle and an optional center cross. Wire it after the sampler to check
the composition, then bypass it for the final save.
"""

import numpy as np
import torch
from PIL import Image, ImageDraw


def draw_safe_area(pil, safe_margin=0.1, show_thirds=True, show_center=True,
                   line_width=2):
    """Draw the overlay onto a copy. Pure function for tests."""
    try:
        margin = float(safe_margin)
    except (TypeError, ValueError):
        margin = 0.1
    margin = max(0.0, min(0.25, margin))
    try:
        lw = max(1, int(line_width))
    except (TypeError, ValueError):
        lw = 2
    canvas = pil.convert("RGB")
    w, h = canvas.size
    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    white = (255, 255, 255, 200)
    grey = (255, 255, 255, 110)

    if show_thirds:
        for frac in (1 / 3, 2 / 3):
            x = int(round(w * frac))
            draw.line([(x, 0), (x, h)], fill=grey, width=max(1, lw - 1))
            y = int(round(h * frac))
            draw.line([(0, y), (w, y)], fill=grey, width=max(1, lw - 1))

    mx, my = int(round(w * margin)), int(round(h * margin))
    draw.rectangle((mx, my, w - mx, h - my), outline=white, width=lw)

    if show_center:
        cx, cy = w // 2, h // 2
        arm = max(8, min(w, h) // 24)
        draw.line([(cx - arm, cy), (cx + arm, cy)], fill=white, width=lw)
        draw.line([(cx, cy - arm), (cx, cy + arm)], fill=white, width=lw)

    return Image.alpha_composite(canvas.convert("RGBA"), overlay).convert("RGB")


class FeiFeiSafeAreaOverlay:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "safe_margin": ("FLOAT", {
                    "default": 0.1, "min": 0.0, "max": 0.25, "step": 0.01,
                    "tooltip": "Action-safe inset as a fraction of each side. 0.1 keeps titles clear of the cinematic crop.",
                }),
                "show_thirds": ("BOOLEAN", {"default": True}),
                "show_center": ("BOOLEAN", {"default": True}),
                "line_width": ("INT", {
                    "default": 2, "min": 1, "max": 16, "step": 1,
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "overlay"
    CATEGORY = "FeiFei"

    def overlay(self, image, safe_margin, show_thirds, show_center, line_width):
        if not isinstance(image, torch.Tensor):
            raise ValueError(f"image must be a ComfyUI IMAGE Tensor [B,H,W,C], got {type(image)}")
        if image.ndim == 3:
            image = image.unsqueeze(0)
        if image.ndim != 4:
            raise ValueError(f"Bad image dims, expected [B,H,W,C], got {tuple(image.shape)}")
        frames = []
        for index in range(image.shape[0]):
            array = np.clip(255.0 * image[index][..., :3].cpu().numpy(), 0, 255).astype(np.uint8)
            out = draw_safe_area(
                Image.fromarray(array), safe_margin=safe_margin,
                show_thirds=bool(show_thirds), show_center=bool(show_center),
                line_width=line_width,
            )
            frames.append(torch.from_numpy(np.asarray(out, dtype=np.float32) / 255.0))
        return (torch.stack(frames).clamp(0, 1).to(image.dtype).contiguous(),)


NODE_CLASS_MAPPINGS = {"FeiFeiSafeAreaOverlay": FeiFeiSafeAreaOverlay}

NODE_DISPLAY_NAME_MAPPINGS = {"FeiFeiSafeAreaOverlay": "Safe Area Overlay"}
