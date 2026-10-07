"""Before/After compare node: grain A/B checks and grade reviews in one frame.

Takes the pre-grade and post-grade images and renders them into a single
frame: side-by-side, stacked, or a wipe split with a divider line. The second
image is fitted to the first, so a resized upscale still compares cleanly.
"""

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

COMPARE_MODES = [
    "Side by Side (H)",
    "Stack (V)",
    "Wipe (Left-Right)",
    "Wipe (Top-Bottom)",
]

MODE_SIDE, MODE_STACK, MODE_WIPE_H, MODE_WIPE_V = COMPARE_MODES


def _fit_after(before_pil, after_pil):
    """Fit the after image onto the before size; passthrough when equal."""
    if after_pil.size == before_pil.size:
        return after_pil
    return after_pil.resize(before_pil.size, Image.LANCZOS)


def _draw_label(draw, text, xy, size):
    """Small corner tag with the PIL default font; never fails the frame."""
    if not isinstance(text, str) or not text.strip():
        return
    try:
        font = ImageFont.load_default(size=size)
    except Exception:
        font = ImageFont.load_default()
    x, y = xy
    pad = max(4, size // 4)
    left, top, right, bottom = draw.textbbox((x, y), text, font=font)
    draw.rectangle(
        (left - pad, top - pad, right + pad, bottom + pad), fill=(0, 0, 0, 170)
    )
    draw.text((x, y), text, font=font, fill=(255, 255, 255, 255))


def compare_pils(before_pil, after_pil, mode=MODE_SIDE, divider=0.5,
                 line_width=2, label_before="Before", label_after="After",
                 show_labels=True):
    """Render one compare frame from two PIL images. Pure function for tests."""
    before = before_pil.convert("RGB")
    after = _fit_after(before, after_pil.convert("RGB"))
    try:
        pos = float(divider)
    except (TypeError, ValueError):
        pos = 0.5
    pos = max(0.0, min(1.0, pos))
    try:
        lw = max(1, int(line_width))
    except (TypeError, ValueError):
        lw = 2
    w, h = before.size

    if mode == MODE_STACK:
        canvas = Image.new("RGB", (w, h * 2), (0, 0, 0))
        canvas.paste(before, (0, 0))
        canvas.paste(after, (0, h))
        if lw > 0:
            d = ImageDraw.Draw(canvas)
            d.rectangle((0, h - lw // 2, w, h + lw // 2), fill=(255, 255, 255))
    elif mode == MODE_WIPE_H:
        canvas = before.copy()
        split = int(round(w * pos))
        canvas.paste(after.crop((split, 0, w, h)), (split, 0))
        if lw > 0:
            d = ImageDraw.Draw(canvas)
            d.rectangle((split - lw // 2, 0, split + lw // 2, h), fill=(255, 255, 255))
    elif mode == MODE_WIPE_V:
        canvas = before.copy()
        split = int(round(h * pos))
        canvas.paste(after.crop((0, split, w, h)), (0, split))
        if lw > 0:
            d = ImageDraw.Draw(canvas)
            d.rectangle((0, split - lw // 2, w, split + lw // 2), fill=(255, 255, 255))
    else:  # MODE_SIDE default
        canvas = Image.new("RGB", (w * 2, h), (0, 0, 0))
        canvas.paste(before, (0, 0))
        canvas.paste(after, (w, 0))
        if lw > 0:
            d = ImageDraw.Draw(canvas)
            d.rectangle((w - lw // 2, 0, w + lw // 2, h), fill=(255, 255, 255))

    if show_labels:
        d = ImageDraw.Draw(canvas, "RGBA")
        tag = max(12, min(canvas.size) // 40)
        if mode == MODE_STACK:
            _draw_label(d, label_before, (8, 8), tag)
            _draw_label(d, label_after, (8, h + 8), tag)
        elif mode in (MODE_WIPE_H, MODE_WIPE_V):
            _draw_label(d, label_before, (8, 8), tag)
            _draw_label(d, label_after, (canvas.size[0] - 8 - tag * 4, 8), tag)
        else:
            _draw_label(d, label_before, (8, 8), tag)
            _draw_label(d, label_after, (w + 8, 8), tag)
    return canvas


class FeiFeiBeforeAfterCompare:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "before": ("IMAGE",),
                "after": ("IMAGE",),
                "mode": (COMPARE_MODES, {"default": MODE_SIDE}),
                "divider": ("FLOAT", {
                    "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Wipe split position. Only read in the two Wipe modes.",
                }),
                "line_width": ("INT", {
                    "default": 2, "min": 0, "max": 16, "step": 1,
                    "tooltip": "Divider line width in pixels. 0 hides the line.",
                }),
                "show_labels": ("BOOLEAN", {"default": True}),
                "label_before": ("STRING", {"multiline": False, "default": "Before"}),
                "label_after": ("STRING", {"multiline": False, "default": "After"}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "compare"
    CATEGORY = "FeiFei"

    def compare(self, before, after, mode, divider, line_width,
                show_labels, label_before, label_after):
        for name, value in (("before", before), ("after", after)):
            if not isinstance(value, torch.Tensor):
                raise ValueError(f"{name} must be a ComfyUI IMAGE Tensor [B,H,W,C], got {type(value)}")
            if value.ndim not in (3, 4):
                raise ValueError(f"Bad {name} dims, expected [B,H,W,C], got {tuple(value.shape)}")
        b = before.unsqueeze(0) if before.ndim == 3 else before
        a = after.unsqueeze(0) if after.ndim == 3 else after
        count = max(b.shape[0], a.shape[0])
        frames = []
        for index in range(count):
            b_arr = np.clip(255.0 * b[index % b.shape[0]][..., :3].cpu().numpy(), 0, 255).astype(np.uint8)
            a_arr = np.clip(255.0 * a[index % a.shape[0]][..., :3].cpu().numpy(), 0, 255).astype(np.uint8)
            canvas = compare_pils(
                Image.fromarray(b_arr), Image.fromarray(a_arr),
                mode=mode, divider=divider, line_width=line_width,
                label_before=label_before, label_after=label_after,
                show_labels=bool(show_labels),
            )
            frames.append(torch.from_numpy(np.asarray(canvas, dtype=np.float32) / 255.0))
        return (torch.stack(frames).clamp(0, 1).to(before.dtype).contiguous(),)


NODE_CLASS_MAPPINGS = {"FeiFeiBeforeAfterCompare": FeiFeiBeforeAfterCompare}

NODE_DISPLAY_NAME_MAPPINGS = {"FeiFeiBeforeAfterCompare": "Before / After Compare"}
