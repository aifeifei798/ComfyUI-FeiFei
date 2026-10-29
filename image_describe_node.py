"""Image describe node: upload one image, let a vision model behind an
OpenAI-compatible API describe it, and return that text as a single string.

Deliberately plain: no JSON, no language switch, no extra outputs. Whatever the
model writes is the answer, minus any <think>...</think> reasoning block it
leaves in the reply, so the output can be wired straight into a prompt box.

api_base must serve a vision model (Qwen-VL, MiniCPM-V, LFM-VL, ...); llama.cpp's
native /completion endpoint has no image support, so this node only uses
/v1/chat/completions.
"""

import os

from .llm_common import (
    _coerce_text,
    _image_file_to_data_url,
    _post_chat_completions,
    _strip_thinking,
    resolve_endpoint,
)

# Longest-side cap before upload, keeps the base64 payload small
MAX_IMAGE_SIDE = 1024

DEFAULT_INSTRUCTION = (
    "Describe this image in detail: subject, action, scene, lighting and style. "
    "Plain prose in one paragraph, no lists, no JSON, no preamble."
)

SYSTEM_PROMPT = (
    "You are an expert image analyst. Answer with the description itself only; "
    "no reasoning, no summary of the task, no alternatives."
)


class FeiFeiImageDescribe:
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
                "image": (sorted(files), {"image_upload": True}),
                "instruction": ("STRING", {"multiline": True, "default": DEFAULT_INSTRUCTION}),
                "temperature": ("FLOAT", {"default": 0.3, "min": 0.1, "max": 1.5, "step": 0.05}),
                "max_tokens": ("INT", {"default": 1024, "min": 64, "max": 8192, "step": 64}),
                "max_side": ("INT", {"default": MAX_IMAGE_SIDE, "min": 256, "max": 4096, "step": 128}),
            },
            "optional": {
                "timeout": ("INT", {
                    "default": 180, "min": 10, "max": 600, "step": 5,
                    "tooltip": "Request timeout in seconds.",
                }),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("description",)
    FUNCTION = "describe_image"
    CATEGORY = "FeiFei"

    def describe_image(self, image, instruction, temperature, max_tokens, max_side=MAX_IMAGE_SIDE, timeout=180):
        try:
            import folder_paths
            image_path = folder_paths.get_annotated_filepath(image)
        except Exception:
            image_path = image if isinstance(image, str) and os.path.isfile(image) else None
        if not image_path or not os.path.isfile(image_path):
            return (f"API Error: image file not found: {image}",)

        api_base, api_key, model = resolve_endpoint("FeiFeiImageDescribe")
        base = (api_base or "").strip().rstrip("/")
        if not base:
            return ("API Error: api_base is empty in config.json",)

        try:
            data_url = _image_file_to_data_url(image_path, max_side)
        except Exception as e:
            return (f"API Error: failed to read/encode image: {e}",)

        payload = {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": [
                    {"type": "text", "text": instruction or DEFAULT_INSTRUCTION},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ]},
            ],
            "temperature": temperature,
            "max_tokens": int(max_tokens),
            "stream": False,
            "enable_thinking": False,
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
            if not choices or not isinstance(choices[0], dict):
                raise ValueError("LLM response missing choices")
            message = choices[0].get("message") or {}
            text = _strip_thinking(_coerce_text(message.get("content", "")))
        except Exception as e:
            return (f"API Error: {e}",)

        if not text:
            return ("API Error: model returned an empty reply",)
        return (text,)


NODE_CLASS_MAPPINGS = {"FeiFeiImageDescribe": FeiFeiImageDescribe}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FeiFeiImageDescribe": "Image Describe (API: edit config.json)",
}
