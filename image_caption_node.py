"""Image caption node: upload an image and let a vision model write Chinese and
English prompts for it.

`source` picks where that model runs: the OpenAI-compatible endpoint in
config.json (which must serve a vision model; llama.cpp's native /completion has
no image support, so only /v1/chat/completions is used) or the HF-format folder
named in config.json, loaded in-process.
"""

import os

from .llm_common import (
    SOURCE_API,
    SOURCE_LOCAL,
    SOURCE_MODES,
    THINKING_OURS,
    THINKING_MODEL,
    THINKING_BOTH,
    THINKING_MODES,
    _coerce_text,
    _open_image_rgb,
    _image_file_to_data_url,
    _post_chat_completions,
    _extract_json_object,
    resolve_endpoint,
)
from .local_model import run_chat

DEFAULT_INSTRUCTION = (
    "Look at this image carefully and output ONLY one JSON object, nothing else: "
    '{"chinese": "describe the image content, subject, action, scene, lighting and style in detail in Chinese", '
    '"english": "English image-generation prompt, comma-separated tags and quality words, '
    "directly usable in Stable Diffusion or Qwen-Image, e.g. '1girl, ... , masterpiece, best quality'\"}"
)

# Longest-side cap before upload, keeps payloads and tokens small
MAX_IMAGE_SIDE = 1024

SYSTEM_PROMPT = "You are an expert image analyst. Always respond with valid JSON only."

# Minimal instruction for the model-native chain: JSON shape only, no scaffolding
NATIVE_INSTRUCTION = (
    "You are an expert image analyst. Think freely about the image, "
    "then output ONLY one valid JSON object, no other text: "
    '{"chinese": "<detailed description of the image in Chinese>", '
    '"english": "<English image-generation prompt, comma-separated tags>"}.'
)


def _content_to_text(content):
    """Accept content returned either as a string or as a list of parts"""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(parts).strip()
    return ""


class FeiFeiImageCaptioner:
    @classmethod
    def INPUT_TYPES(cls):
        try:
            import folder_paths
            input_dir = folder_paths.get_input_directory()
            files = [f for f in os.listdir(input_dir)
                     if os.path.isfile(os.path.join(input_dir, f))]
            files = folder_paths.filter_files_content_types(files, ["image"])
        except Exception:
            files = []
        return {
            "required": {
                "source": (SOURCE_MODES, {
                    "default": SOURCE_API,
                    "tooltip": "API (config.json) posts the image to the endpoint in config.json; "
                               "Local HF model runs the `model_dir` folder listed in config.json with transformers.",
                }),
                "image": (sorted(files), {"image_upload": True}),
                "instruction": ("STRING", {"multiline": True, "default": DEFAULT_INSTRUCTION}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.1, "max": 1.5, "step": 0.05}),
                "max_tokens": ("INT", {"default": 1024, "min": 64, "max": 8192, "step": 64}),
                "max_side": ("INT", {"default": MAX_IMAGE_SIDE, "min": 256, "max": 4096, "step": 128}),
                "thinking_mode": (THINKING_MODES, {"default": THINKING_OURS}),
            },
            "optional": {
                "timeout": ("INT", {
                    "default": 180, "min": 10, "max": 600, "step": 5,
                    "tooltip": "API mode only: request timeout in seconds.",
                }),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("chinese", "english", "thinking")
    FUNCTION = "caption_image"
    CATEGORY = "FeiFei"

    def caption_image(self, source, image, instruction, temperature, max_tokens,
                      max_side=MAX_IMAGE_SIDE, thinking_mode=THINKING_OURS, timeout=180):
        try:
            import folder_paths
            image_path = folder_paths.get_annotated_filepath(image)
        except Exception:
            image_path = image if isinstance(image, str) and os.path.isfile(image) else None
        if not image_path or not os.path.isfile(image_path):
            return (f"API Error: image file not found: {image}", "", "")

        # Model native ignores the instruction box and uses the built-in minimal
        # instruction so the model's own thinking runs; Ours/Both use the box
        if thinking_mode == THINKING_MODEL:
            effective_instruction = NATIVE_INSTRUCTION
        else:
            effective_instruction = instruction or DEFAULT_INSTRUCTION
        enable_thinking = thinking_mode in (THINKING_MODEL, THINKING_BOTH)

        if source == SOURCE_LOCAL:
            raw, thinking, error = self._caption_local(
                image_path, effective_instruction, temperature, max_tokens, max_side, enable_thinking)
        else:
            raw, thinking, error = self._caption_api(
                image_path, effective_instruction, temperature, max_tokens, max_side, enable_thinking, timeout)
        if error:
            return (error, "", "")

        parsed = _extract_json_object(raw)
        if parsed is not None and (parsed.get("chinese") or parsed.get("english")):
            chinese = str(parsed.get("chinese", "") or "").strip()
            english = str(parsed.get("english", "") or "").strip()
            return (chinese, english, thinking)

        print("[FeiFeiImageCaptioner] model did not reply in JSON, full text goes to chinese.")
        return (raw.strip(), "", thinking)

    def _caption_local(self, image_path, instruction, temperature, max_tokens, max_side, enable_thinking):
        try:
            image = _open_image_rgb(image_path, max_side)
        except Exception as e:
            return "", "", f"Local Error: failed to read image: {e}"
        content = [
            {"type": "image", "image": image},
            {"type": "text", "text": instruction},
        ]
        return run_chat("FeiFeiImageCaptioner", SYSTEM_PROMPT, content, temperature,
                        enable_thinking, max_tokens)

    def _caption_api(self, image_path, instruction, temperature, max_tokens, max_side,
                     enable_thinking, timeout):
        api_base, api_key, model = resolve_endpoint("FeiFeiImageCaptioner")
        base = (api_base or "").strip().rstrip("/")
        if not base:
            return "", "", "API Error: api_base is empty in config.json"

        try:
            data_url = _image_file_to_data_url(image_path, max_side)
        except Exception as e:
            return "", "", f"API Error: failed to read/encode image: {e}"

        payload = {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": [
                    {"type": "text", "text": instruction},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ]},
            ],
            "temperature": temperature,
            "max_tokens": int(max_tokens),
            "stream": False,
            "enable_thinking": enable_thinking,
        }
        model_name = (model or "").strip()
        if model_name:
            payload["model"] = model_name

        try:
            seconds = int(timeout)
        except (TypeError, ValueError):
            seconds = 180
        seconds = max(10, min(600, seconds))
        try:
            res_json = _post_chat_completions(base, payload, timeout=seconds, api_key=api_key)
            choices = res_json.get("choices") if isinstance(res_json, dict) else None
            raw = ""
            thinking = ""
            if choices and isinstance(choices[0], dict):
                message = choices[0].get("message", {})
                raw = _content_to_text(message.get("content", ""))
                thinking = _coerce_text(message.get("reasoning_content", ""))
            if not raw:
                raise ValueError("LLM response missing text content")
        except Exception as e:
            return "", "", f"API Error: {e}"
        return raw, thinking, ""


NODE_CLASS_MAPPINGS = {"FeiFeiImageCaptioner": FeiFeiImageCaptioner}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FeiFeiImageCaptioner": "Image Captioner (API / Local HF)",
}
