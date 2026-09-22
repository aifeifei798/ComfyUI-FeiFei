
from .qwen_prompt_node import QwenImagePromptEnhancer

NODE_CLASS_MAPPINGS = {
    "QwenImagePromptEnhancer": QwenImagePromptEnhancer
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "QwenImagePromptEnhancer": "Qwen-Image Prompt Enhancer (LLaMA)"
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']