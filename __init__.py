import traceback


NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}


def _register(mappings, display_mappings):
    NODE_CLASS_MAPPINGS.update(mappings)
    NODE_DISPLAY_NAME_MAPPINGS.update(display_mappings)


# Import nodes one by one: a single node failing must not kill the whole pack
try:
    from .qwen_prompt_node import QwenImagePromptEnhancer
    _register(
        {"QwenImagePromptEnhancer": QwenImagePromptEnhancer},
        {"QwenImagePromptEnhancer": "Qwen-Image Prompt Enhancer (LLaMA)"},
    )
except Exception:
    print("[FeiFei] QwenImagePromptEnhancer failed to load:")
    traceback.print_exc()

try:
    from .watermark_node import WatermarkNode
    _register(
        {"WatermarkNode": WatermarkNode},
        {"WatermarkNode": "Watermark"},
    )
except Exception:
    print("[FeiFei] WatermarkNode failed to load:")
    traceback.print_exc()

try:
    from .film_grain_node import FeiFeiFilmGrainTone
    _register(
        {"FeiFeiFilmGrainTone": FeiFeiFilmGrainTone},
        {"FeiFeiFilmGrainTone": "FeiFei Film Grain & Tone"},
    )
except Exception:
    print("[FeiFei] FeiFeiFilmGrainTone failed to load:")
    traceback.print_exc()

try:
    from .image_to_rgb import ImageToRGB
    _register(
        {"ImageToRGB": ImageToRGB},
        {"ImageToRGB": "Image To RGB (Force 3-Channel)"},
    )
except Exception:
    print("[FeiFei] ImageToRGB failed to load:")
    traceback.print_exc()

try:
    from .style_selector_node_zh_ex import StyleSelectorNodeZhex
    _register(
        {"StyleSelectorNodeZhex": StyleSelectorNodeZhex},
        {"StyleSelectorNodeZhex": "Style Selector EX"},
    )
except Exception:
    print("[FeiFei] StyleSelectorNodeZhex failed to load:")
    traceback.print_exc()

try:
    from .ComfyUI_SaveWebP.save_webp_node import (
        SaveWebPWithTimestamp,
        LoadWebPInfo,
    )
    _register(
        {
            "SaveWebPWithTimestamp": SaveWebPWithTimestamp,
            "LoadWebPInfo": LoadWebPInfo,
        },
        {
            "SaveWebPWithTimestamp": "Save WebP (Timestamp)",
            "LoadWebPInfo": "Load WebP Info",
        },
    )
except Exception:
    print("[FeiFei] SaveWebPWithTimestamp failed to load:")
    traceback.print_exc()

try:
    from .aspect_ratio_node import FeiFeiAspectRatio
    _register(
        {"FeiFeiAspectRatio": FeiFeiAspectRatio},
        {"FeiFeiAspectRatio": "Aspect Ratio (1024)"},
    )
except Exception:
    print("[FeiFei] FeiFeiAspectRatio failed to load:")
    traceback.print_exc()

try:
    from .prompt_director_node import FeiFeiPromptDirector
    _register(
        {"FeiFeiPromptDirector": FeiFeiPromptDirector},
        {"FeiFeiPromptDirector": "Prompt Director"},
    )
except Exception:
    print("[FeiFei] FeiFeiPromptDirector failed to load:")
    traceback.print_exc()

try:
    from .image_caption_node import FeiFeiImageCaptioner
    _register(
        {"FeiFeiImageCaptioner": FeiFeiImageCaptioner},
        {"FeiFeiImageCaptioner": "Image Captioner"},
    )
except Exception:
    print("[FeiFei] FeiFeiImageCaptioner failed to load:")
    traceback.print_exc()

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']