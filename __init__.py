
import traceback


NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}


def _register(mappings, display_mappings):
    NODE_CLASS_MAPPINGS.update(mappings)
    NODE_DISPLAY_NAME_MAPPINGS.update(display_mappings)


# 逐个导入：单个节点失败不拖死整个包，ComfyUI 启动日志可见警告
try:
    from .qwen_prompt_node import QwenImagePromptEnhancer
    _register(
        {"QwenImagePromptEnhancer": QwenImagePromptEnhancer},
        {"QwenImagePromptEnhancer": "Qwen-Image Prompt Enhancer (LLaMA)"},
    )
except Exception:
    print("[FeiFei] QwenImagePromptEnhancer 加载失败：")
    traceback.print_exc()

try:
    from .watermark_node import WatermarkNode
    _register(
        {"WatermarkNode": WatermarkNode},
        {"WatermarkNode": "图像水印 (Watermark)"},
    )
except Exception:
    print("[FeiFei] WatermarkNode 加载失败：")
    traceback.print_exc()

try:
    from .image_to_rgb import ImageToRGB
    _register(
        {"ImageToRGB": ImageToRGB},
        {"ImageToRGB": "Image To RGB (Force 3-Channel)"},
    )
except Exception:
    print("[FeiFei] ImageToRGB 加载失败：")
    traceback.print_exc()

try:
    from .style_selector_node_zh_ex import StyleSelectorNodeZhex
    _register(
        {"StyleSelectorNodeZhex": StyleSelectorNodeZhex},
        {"StyleSelectorNodeZhex": "风格选择器扩展版"},
    )
except Exception:
    print("[FeiFei] StyleSelectorNodeZhex 加载失败：")
    traceback.print_exc()

try:
    from .ComfyUI_SaveWebP.save_webp_node import (
        SaveWebPWithTimestamp,
    )
    _register(
        {"SaveWebPWithTimestamp": SaveWebPWithTimestamp},
        {"SaveWebPWithTimestamp": "Save WebP (Timestamp)"},
    )
except Exception:
    print("[FeiFei] SaveWebPWithTimestamp 加载失败：")
    traceback.print_exc()

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']