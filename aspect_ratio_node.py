"""宽高比尺寸节点：锁定一边为基准边，另一边按比例推算，对齐到 16 的倍数。

替代官方 Resolution Selector（按 megapixels 反推）的心算，专为
“宽 1024 的竖构图 / 高 1024 的横构图”这类固定短边工作流设计。
"""

import re

ASPECT_RATIOS = [
    "1:1",
    "1:2",
    "2:1",
    "3:4",
    "4:3",
    "9:16",
    "16:9",
    "9:21",
    "21:9",
]

LOCK_SHORT_SIDE = "Short Side (recommended)"
LOCK_WIDTH = "Fixed Width (width=base)"
LOCK_HEIGHT = "Fixed Height (height=base)"

LOCK_MODES = [LOCK_SHORT_SIDE, LOCK_WIDTH, LOCK_HEIGHT]

# Legacy Chinese values from older workflows are mapped to the new ones
LEGACY_LOCK_MODES = {
    "固定短边 (推荐)": LOCK_SHORT_SIDE,
    "固定宽度 (width=基准)": LOCK_WIDTH,
    "固定高度 (height=基准)": LOCK_HEIGHT,
}


def _normalize_lock_mode(lock_mode):
    if lock_mode in LEGACY_LOCK_MODES:
        return LEGACY_LOCK_MODES[lock_mode]
    return lock_mode


def _parse_ratio(ratio_str):
    """解析 "w:h" 返回 (w, h)，非法时返回 (1, 1)"""
    if not isinstance(ratio_str, str):
        return 1, 1
    match = re.match(r"\s*(\d+(?:\.\d+)?)\s*[:：/]\s*(\d+(?:\.\d+)?)\s*$", ratio_str)
    if not match:
        return 1, 1
    w, h = float(match.group(1)), float(match.group(2))
    if w <= 0 or h <= 0:
        return 1, 1
    return w, h


def _align16(value):
    """四舍五入到 16 的倍数，最小 16"""
    return max(16, int(round(value / 16.0)) * 16)


def calc_size(ratio_str, lock_mode, base_side):
    """核心计算：返回 (width, height)，纯函数便于单测"""
    lock_mode = _normalize_lock_mode(lock_mode)
    if not isinstance(base_side, (int, float)):
        base_side = 1024
    base = _align16(int(base_side))
    w_ratio, h_ratio = _parse_ratio(ratio_str)

    if lock_mode == LOCK_HEIGHT or (
        lock_mode == LOCK_SHORT_SIDE and w_ratio > h_ratio
    ):
        # 固定高度：横构图家族（16:9 / 21:9 / 4:3 / 2:1），高度=基准
        height = base
        width = _align16(base * w_ratio / h_ratio)
    elif lock_mode == LOCK_WIDTH or (
        lock_mode == LOCK_SHORT_SIDE and w_ratio < h_ratio
    ):
        # 固定宽度：竖构图家族（9:16 / 9:21 / 3:4 / 1:2），宽度=基准
        width = base
        height = _align16(base * h_ratio / w_ratio)
    else:
        # 1:1 或无法判断：双边=基准
        width = base
        height = base
    return width, height


class FeiFeiAspectRatio:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "aspect_ratio": (ASPECT_RATIOS, {"default": "9:16"}),
                "lock_mode": (LOCK_MODES, {"default": LOCK_SHORT_SIDE}),
                "base_side": (
                    "INT",
                    {"default": 1024, "min": 16, "max": 8192, "step": 16},
                ),
            }
        }

    RETURN_TYPES = ("INT", "INT", "STRING")
    RETURN_NAMES = ("width", "height", "ratio")
    FUNCTION = "get_size"
    CATEGORY = "FeiFei"

    def get_size(self, aspect_ratio, lock_mode, base_side):
        width, height = calc_size(aspect_ratio, lock_mode, base_side)
        return (width, height, aspect_ratio)


NODE_CLASS_MAPPINGS = {"FeiFeiAspectRatio": FeiFeiAspectRatio}

NODE_DISPLAY_NAME_MAPPINGS = {"FeiFeiAspectRatio": "Aspect Ratio (1024)"}
