"""Negative-prompt library node: one dropdown instead of a pasted tag wall.

Picks a curated negative stack, appends an optional custom line and an
optional upstream string, then dedupes commas the same way Style Selector
does. No network, no models.
"""

import re

NEGATIVE_PRESETS = {
    "(None)": "",
    "SDXL Base": (
        "lowres, bad anatomy, bad hands, missing fingers, extra fingers, "
        "fused fingers, mutated hands, blurry, worst quality, low quality, "
        "jpeg artifacts, watermark, signature, text, logo"
    ),
    "Photoreal Skin": (
        "waxy skin, plastic skin, over-smoothed, cartoon, illustration, "
        "3d render, deformed, disfigured, bad anatomy, blurry, lowres, "
        "watermark, text, logo"
    ),
    "Anime Clean": (
        "lowres, bad anatomy, bad hands, missing limbs, blurry, worst quality, "
        "low quality, jpeg artifacts, watermark, signature, username, text"
    ),
    "Hands & Faces": (
        "bad hands, malformed hands, extra fingers, missing fingers, fused "
        "fingers, deformed face, asymmetric eyes, crossed eyes, bad anatomy"
    ),
    "Text & Logo Free": "text, watermark, logo, signature, username, subtitle, caption",
    "Blur & Noise Free": "blurry, out of focus, noisy, grainy, jpeg artifacts, lowres, low quality",
}

PRESET_NAMES = list(NEGATIVE_PRESETS.keys())


def build_negative(preset, custom="", extra=""):
    """Assemble the negative prompt. Pure function for tests."""
    parts = []
    base = NEGATIVE_PRESETS.get(preset, "")
    if isinstance(base, str) and base.strip():
        parts.append(base.strip())
    for chunk in (custom, extra):
        if isinstance(chunk, str) and chunk.strip():
            parts.append(chunk.strip())
    text = ", ".join(parts).strip().strip(",").strip()
    text = re.sub(r",\s*,", ",", text)
    text = re.sub(r"\s+", " ", text)
    text = text.strip().strip(",").strip()
    # Ordered dedupe of comma-separated terms (case-insensitive), so a custom
    # line repeating a preset term does not stack it twice
    seen, unique = set(), []
    for term in (t.strip() for t in text.split(",") if t.strip()):
        key = term.lower()
        if key not in seen:
            seen.add(key)
            unique.append(term)
    return ", ".join(unique)


class FeiFeiNegativeLibrary:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "preset": (PRESET_NAMES, {"default": "SDXL Base"}),
                "custom": ("STRING", {"multiline": True, "default": ""}),
            },
            "optional": {
                "extra": ("STRING", {"forceInput": True}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("negative_prompt",)
    FUNCTION = "build"
    CATEGORY = "FeiFei"

    def build(self, preset, custom, extra=None):
        return (build_negative(preset, custom, extra),)


NODE_CLASS_MAPPINGS = {"FeiFeiNegativeLibrary": FeiFeiNegativeLibrary}

NODE_DISPLAY_NAME_MAPPINGS = {"FeiFeiNegativeLibrary": "Negative Library"}
