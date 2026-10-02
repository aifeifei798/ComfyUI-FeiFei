"""Cinematic still node: theatrical crop plus a film subtitle.

Turns a generated image into a "movie screenshot": the picture is scaled to
fill the chosen aspect and the overflow is cropped off the centre, then the
line is typeset like a real film subtitle, with an optional dashed timecode
overlay.

Aspect ratio, lock mode and base side come from the Aspect Ratio (1024) node,
so one ratio drives both nodes and the frames come out the same size.

Cinematic Crop fills the frame edge to edge, so a 16:9 render in 21:9 loses its
top and bottom rather than gaining side bars. Letterbox is the only mode that
fits the whole image and bars the leftover axis.
"""

import os

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from .aspect_ratio_node import ASPECT_RATIOS, LOCK_MODES, LOCK_SHORT_SIDE, calc_size


# The ratio list and the sizing math come from the Aspect Ratio (1024) node, so
# one workflow can feed both nodes the same ratio and get the same frame.
FILL_MODES = ["Cinematic Crop (fill frame)", "Letterbox (black bars)"]
FILL_CROP, FILL_PAD = FILL_MODES

SUBTITLE_POSITIONS = ["Bottom", "Center"]

SUBTITLE_COLORS = ["Yellow", "White", "Amber"]
_SUBTITLE_RGB = {
    "Yellow": (245, 214, 90),
    "White": (246, 246, 243),
    "Amber": (232, 166, 62),
}

# User path first, then a Latin-capable font. DejaVu is deliberately ahead of
# the CJK faces: it is the one file that is present on nearly every Linux box.
FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "C:/Windows/Fonts/arial.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
]

# Only consulted when the line actually contains CJK, because DejaVu renders
# those characters as tofu boxes.
CJK_FONT_CANDIDATES = [
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    "C:/Windows/Fonts/msyh.ttc",
    "/System/Library/Fonts/PingFang.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
]

SUBTITLE_MAX_WIDTH = 0.86  # of the picture width
LINE_SPACING = 1.24  # baseline step over one line height
SECONDARY_SCALE = 0.82  # the English line under Chinese is set smaller


def _resolve_font_path(font_path, needs_cjk=False):
    """Return a usable font path, or None when every candidate is missing. A CJK
    line needs a face that actually carries those glyphs, so the chain changes."""
    user = font_path.strip() if isinstance(font_path, str) else ""
    if user and os.path.isfile(user):
        return user
    candidates = list(CJK_FONT_CANDIDATES) + FONT_CANDIDATES if needs_cjk else FONT_CANDIDATES
    for cand in candidates:
        if os.path.isfile(cand):
            return cand
    return None


def _needs_cjk(*texts):
    return any(_is_wide(ch) for text in texts if isinstance(text, str) for ch in text)


# A batch draws the same two faces on every frame, and sweeping subtitle_size
# would otherwise pile up a face per step, so the cache is bounded.
_FONT_CACHE = {}
_FONT_CACHE_MAX = 8


def _load_font(path, size):
    """Cached per (path, size). Truncates to the PIL default font when the file
    cannot be opened."""
    key = (path, size)
    if key not in _FONT_CACHE:
        font = None
        if path:
            try:
                font = ImageFont.truetype(path, size)
            except (IOError, OSError) as e:
                print(f"Warning: failed to load font {path} ({e}), using the PIL default font.")
        if len(_FONT_CACHE) >= _FONT_CACHE_MAX:
            _FONT_CACHE.clear()
        _FONT_CACHE[key] = font or ImageFont.load_default()
    return _FONT_CACHE[key]


def _fit_into(pil, width, height, fill_mode):
    """Scale the picture into the frame. Returns the frame-sized RGB image and
    the (x, y, w, h) rectangle the picture actually covers.

    Cinematic Crop fills the frame edge to edge and cuts the overflow off the
    centre, which is what makes a still read as a real movie frame: a 16:9 render
    in 21:9 loses its top and bottom, never its sides. Letterbox is the only mode
    that leaves black bars, on whichever axis is left over.
    """
    sw, sh = pil.size
    if fill_mode == FILL_PAD:
        scale = min(width / sw, height / sh)
        resized = pil.resize(
            (max(1, round(sw * scale)), max(1, round(sh * scale))), Image.LANCZOS
        )
        padded = Image.new("RGB", (width, height), (0, 0, 0))
        x, y = (width - resized.width) // 2, (height - resized.height) // 2
        padded.paste(resized, (x, y))
        return padded, (x, y, resized.width, resized.height)

    scale = max(width / sw, height / sh)
    resized = pil.resize(
        (max(1, round(sw * scale)), max(1, round(sh * scale))), Image.LANCZOS
    )
    left, top = (resized.width - width) // 2, (resized.height - height) // 2
    return resized.crop((left, top, left + width, top + height)), (0, 0, width, height)


def _is_wide(ch):
    """CJK and other full-width blocks wrap per character, Latin per word"""
    code = ord(ch)
    return (
        0x1100 <= code <= 0x115F
        or 0x2E80 <= code <= 0xA4CF
        or 0xAC00 <= code <= 0xD7A3
        or 0xF900 <= code <= 0xFAFF
        or 0xFE30 <= code <= 0xFE6F
        or 0xFF00 <= code <= 0xFF60
        or 0xFFE0 <= code <= 0xFFE6
    )


def _tokenize(text):
    tokens, buf = [], ""
    for ch in text:
        if _is_wide(ch):
            if buf:
                tokens.append(buf)
                buf = ""
            tokens.append(ch)
        elif ch.isspace():
            if buf:
                tokens.append(buf)
                buf = ""
            tokens.append(" ")
        else:
            buf += ch
    if buf:
        tokens.append(buf)
    return tokens


def _text_width(draw, text, font):
    left, _, right, _ = draw.textbbox((0, 0), text, font=font)
    return right - left


def _wrap(draw, text, font, max_width):
    """Greedy wrap; newlines in the input stay hard breaks"""
    lines = []
    for paragraph in str(text).splitlines():
        if not paragraph.strip():
            continue
        current = ""
        for token in _tokenize(paragraph):
            trial = (current + token).strip()
            if current and _text_width(draw, trial, font) > max_width:
                lines.append(current.strip())
                current = "" if token == " " else token
            else:
                current += token
        if current.strip():
            lines.append(current.strip())
    return lines


def _layout_lines(draw, texts, max_width):
    """Wrap every text against its own font. Returns (line, font, size, ascent,
    descent) tuples in draw order."""
    lines = []
    for text, font, size in texts:
        if not isinstance(text, str) or not text.strip():
            continue
        ascent, descent = font.getmetrics()
        for line in _wrap(draw, text, font, max_width):
            lines.append((line, font, size, ascent, descent))
    return lines


def _block_height(lines):
    """Ink height from the top of the first line to the bottom of the last"""
    heights = [asc + desc for _, _, _, asc, desc in lines]
    steps = [round(h * LINE_SPACING) for h in heights]
    return sum(steps) - (steps[-1] - heights[-1])


def _draw_text_block(canvas, lines, color, center_x, bottom_y, shadow):
    """Composite the subtitle: soft black shadow first, crisp glyphs on top"""
    steps = [round((asc + desc) * LINE_SPACING) for _, _, _, asc, desc in lines]
    top = max(0, bottom_y - _block_height(lines))

    layer = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    y = top
    for (text, font, _, _, _), step in zip(lines, steps):
        # Centre on the ink, not on the advance width, so a narrow glyph such as
        # "I" still lands in the middle
        left, _, right, _ = draw.textbbox((0, 0), text, font=font)
        draw.text((center_x - (left + right) / 2, y), text, font=font, fill=color + (255,))
        y += step

    if shadow > 0:
        size = lines[0][2]
        radius = max(1.0, size * 0.06)
        offset = max(1, round(size * 0.06))
        alpha = layer.getchannel("A").filter(ImageFilter.GaussianBlur(radius))
        shifted = Image.new("L", canvas.size, 0)
        shifted.paste(alpha, (0, offset))
        shade = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
        shade.putalpha(shifted.point(lambda a: round(a * shadow)))
        canvas = Image.alpha_composite(canvas, shade)

    return Image.alpha_composite(canvas, layer)


def _dashed_rect(draw, box, dash, gap, width, fill):
    x0, y0, x1, y1 = box

    def hline(xa, xb, y):
        x = xa
        while x < xb:
            draw.line([(x, y), (min(x + dash, xb), y)], fill=fill, width=width)
            x += dash + gap

    def vline(ya, yb, x):
        y = ya
        while y < yb:
            draw.line([(x, y), (x, min(y + dash, yb))], fill=fill, width=width)
            y += dash + gap

    hline(x0, x1, y0)
    hline(x0, x1, y1)
    vline(y0, y1, x0)
    vline(y0, y1, x1)


def _draw_timecode(canvas, text, rect, font_path):
    """Dashed TC box in the top-right corner of the picture, like a camera
    burn-in overlay."""
    rx, ry, rw, rh = rect
    size = max(11, round(min(rw, rh) * 0.026))
    tc_font = _load_font(font_path, size)
    label = text if text.upper().startswith("TC") else f"TC {text}"
    draw = ImageDraw.Draw(canvas)
    left, top, right, bottom = draw.textbbox((0, 0), label, font=tc_font)
    pad = round(size * 0.45)
    margin = max(4, round(rw * 0.02))
    x1 = rx + rw - margin
    y0 = ry + margin
    box = (x1 - (right - left) - pad * 2, y0 + top - pad, x1, y0 + bottom + pad)
    dash = max(2, round(size * 0.35))
    _dashed_rect(draw, box, dash, round(dash * 0.6), max(1, round(size * 0.06)), (235, 235, 235, 220))
    draw.text((box[0] + pad - left, box[1] + pad - top), label, font=tc_font, fill=(240, 240, 240, 235))
    return canvas


def cinematic_frame(
    image,
    aspect_ratio="21:9",
    lock_mode=LOCK_SHORT_SIDE,
    fill_mode=FILL_CROP,
    base_side=1024,
    subtitle_top="",
    subtitle_bottom="",
    subtitle_size=0.045,
    subtitle_color="Yellow",
    shadow=0.8,
    position="Bottom",
    font_path="",
    timecode="",
):
    """Build one cinematic still from a PIL image. Pure function for tests."""
    width, height = calc_size(aspect_ratio, lock_mode, base_side)
    canvas, rect = _fit_into(image.convert("RGB"), width, height, fill_mode)
    pic_x, pic_y, pic_w, pic_h = rect

    path = _resolve_font_path(font_path, _needs_cjk(subtitle_top, subtitle_bottom))
    # Off the short side, not the height: a 9:16 frame is 1024x1824, so keying
    # off the height would set the line almost twice as large as the same still
    # in 21:9, and it would wrap into four lines down a narrow portrait frame.
    size = max(12, round(min(pic_w, pic_h) * subtitle_size))
    canvas = canvas.convert("RGBA")

    if isinstance(timecode, str) and timecode.strip():
        canvas = _draw_timecode(canvas, timecode.strip(), rect, path)

    draw = ImageDraw.Draw(canvas)
    # The second line only shrinks when there is a first line to sit under
    sub_size = round(size * SECONDARY_SCALE) if str(subtitle_top).strip() else size
    lines = _layout_lines(
        draw,
        [
            (subtitle_top, _load_font(path, size), size),
            (subtitle_bottom, _load_font(path, sub_size), sub_size),
        ],
        round(pic_w * SUBTITLE_MAX_WIDTH),
    )
    if lines:
        bottom_bar = height - (pic_y + pic_h)
        if position == "Center":
            bottom_y = pic_y + pic_h // 2 + _block_height(lines) // 2
        elif bottom_bar >= _block_height(lines) + 8:
            # Centred in the letterbox bar, the way a subtitled scope still reads
            bottom_y = pic_y + pic_h + bottom_bar // 2
        else:
            bottom_y = pic_y + pic_h - round(pic_h * 0.04)
        color = _SUBTITLE_RGB.get(subtitle_color, _SUBTITLE_RGB["Yellow"])
        canvas = _draw_text_block(canvas, lines, color, pic_x + pic_w // 2, bottom_y, shadow)

    return canvas.convert("RGB")


class CinematicFrameSubtitle:
    @classmethod
    def INPUT_TYPES(cls):
        # Panel order follows the pipeline: shape the frame, then the line, then
        # the two overlays. The font path stays optional because the fallback
        # chain already covers most installs.
        return {
            "required": {
                "image": ("IMAGE",),
                "aspect_ratio": (ASPECT_RATIOS, {
                    "default": "21:9",
                    "tooltip": "Aspect of the finished still. Same list as the Aspect Ratio (1024) node, so one "
                               "ratio can drive both. 21:9 is the scope look, 16:9 the clean HD frame, 9:16 and "
                               "9:21 the vertical ones.",
                }),
                "lock_mode": (LOCK_MODES, {
                    "default": LOCK_SHORT_SIDE,
                    "tooltip": "Which side base_side pins. Same rule as the Aspect Ratio (1024) node: Short Side "
                               "keeps 21:9 at 1024 high and 9:16 at 1024 wide.",
                }),
                "base_side": ("INT", {
                    "default": 1024, "min": 128, "max": 4096, "step": 16,
                    "tooltip": "The side pinned by lock_mode; the other side follows the ratio. A 21:9 still at "
                               "1024 comes out 2384x1024. Always a multiple of 16.",
                }),
                "fill_mode": (FILL_MODES, {
                    "default": FILL_CROP,
                    "tooltip": "How the picture reaches the frame. Cinematic Crop (the default) scales the image "
                               "to fill the frame and cuts the overflow off the top/bottom or the sides, so the "
                               "frame is always completely covered - a 16:9 render in 21:9 loses its top and "
                               "bottom, never gains side bars. Letterbox is the exception: it shows the whole "
                               "image and adds black bars on the leftover axis.",
                }),
                "subtitle_top": ("STRING", {
                    "multiline": True, "default": "We were all waiting for someone who was never coming.",
                    "tooltip": "Main subtitle line, typeset first and largest. Any language; long lines wrap to "
                               "86% of the picture width.",
                }),
                "subtitle_bottom": ("STRING", {
                    "multiline": True, "default": "",
                    "tooltip": "Second line, set under the main one at 82% of its size. Leave it empty for a "
                               "single-line subtitle.",
                }),
                "subtitle_size": ("FLOAT", {
                    "default": 0.045, "min": 0.015, "max": 0.12, "step": 0.001,
                    "tooltip": "Main line height as a fraction of the frame's short side, so the subtitle keeps "
                               "the same optical size in 21:9 and in 9:16. 0.045 is about 46px on a "
                               "1024px-short-side frame.",
                }),
                "subtitle_color": (SUBTITLE_COLORS, {
                    "default": "Yellow",
                    "tooltip": "Yellow is the usual hard-subtitle look, White the plain theatrical one.",
                }),
                "shadow": ("FLOAT", {
                    "default": 0.8, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Soft drop shadow under the glyphs, scaled to the font size, so the line stays "
                               "readable over a bright frame. 0 turns it off.",
                }),
            },
            "optional": {
                "position": (SUBTITLE_POSITIONS, {
                    "default": "Bottom",
                    "tooltip": "Bottom centres the line in the lower black bar when Letterbox leaves room for "
                               "it, and otherwise sets it just above the bottom edge of the picture. Center is "
                               "the title-card placement.",
                }),
                "timecode": ("STRING", {
                    "multiline": False, "default": "",
                    "tooltip": "Optional burn-in timecode, for example 01:23:45:12, drawn in a dashed box in "
                               "the top-right corner of the picture. Empty skips it.",
                }),
                "font_path": ("STRING", {
                    "multiline": False, "default": "",
                    "tooltip": "Font file for the subtitle and timecode. Empty falls back to DejaVu / Arial for a "
                               "Latin line and to Noto CJK / YaHei / PingFang for a line that needs CJK glyphs.",
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "apply_cinematic_frame"
    CATEGORY = "FeiFei"

    def apply_cinematic_frame(
        self,
        image,
        aspect_ratio,
        lock_mode,
        base_side,
        fill_mode,
        subtitle_top,
        subtitle_bottom,
        subtitle_size,
        subtitle_color,
        shadow,
        position="Bottom",
        timecode="",
        font_path="",
    ):
        if not isinstance(image, torch.Tensor):
            raise ValueError(f"image must be a ComfyUI IMAGE Tensor [B,H,W,C], got {type(image)}")
        if image.ndim == 3:
            image = image.unsqueeze(0)
        if image.ndim != 4:
            raise ValueError(f"Bad image dims, expected [B,H,W,C], got {tuple(image.shape)}")

        # Frame by frame: each batch entry gets the same canvas but its own
        # subtitle, so one full-size RGBA layer is alive at a time.
        frames = []
        for index in range(image.shape[0]):
            array = np.clip(255.0 * image[index][..., :3].cpu().numpy(), 0, 255).astype(np.uint8)
            still = cinematic_frame(
                Image.fromarray(array),
                aspect_ratio=aspect_ratio,
                lock_mode=lock_mode,
                fill_mode=fill_mode,
                base_side=base_side,
                subtitle_top=subtitle_top,
                subtitle_bottom=subtitle_bottom,
                subtitle_size=subtitle_size,
                subtitle_color=subtitle_color,
                shadow=shadow,
                position=position,
                font_path=font_path,
                timecode=timecode,
            )
            frames.append(torch.from_numpy(np.asarray(still, dtype=np.float32) / 255.0))

        return (torch.stack(frames).clamp(0, 1).to(image.dtype).contiguous(),)


NODE_CLASS_MAPPINGS = {"CinematicFrameSubtitle": CinematicFrameSubtitle}

NODE_DISPLAY_NAME_MAPPINGS = {"CinematicFrameSubtitle": "Cinematic Frame & Subtitle"}
