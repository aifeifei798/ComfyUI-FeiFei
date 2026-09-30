"""Image caption node: upload an image and let a vision model behind an
OpenAI-compatible API write Chinese and English prompts for it.

api_base must serve a vision model (Qwen-VL, MiniCPM-V, ...). llama.cpp's native
/completion endpoint has no image support, so this node only uses
/v1/chat/completions.
"""

import base64
import io
import json
import os

from PIL import Image

from .llm_common import (
    THINKING_OURS,
    THINKING_MODEL,
    THINKING_BOTH,
    THINKING_MODES,
    _coerce_text,
    _post_chat_completions,
    _extract_json_object,
    resolve_endpoint,
)

DEFAULT_INSTRUCTION = (
    "Look at this image carefully and output ONLY one JSON object, nothing else: "
    '{"chinese": "describe the image content, subject, action, scene, lighting and style in detail in Chinese", '
    '"english": "English image-generation prompt, comma-separated tags and quality words, '
    "directly usable in Stable Diffusion or Qwen-Image, e.g. '1girl, ... , masterpiece, best quality'\"}"
)

# Longest-side cap before upload, keeps the base64 payload small
MAX_IMAGE_SIDE = 1024


# Minimal instruction for the model-native chain: JSON shape only, no scaffolding
NATIVE_INSTRUCTION = (
    "You are an expert image analyst. Think freely about the image, "
    "then output ONLY one valid JSON object, no other text: "
    '{"chinese": "<detailed description of the image in Chinese>", '
    '"english": "<English image-generation prompt, comma-separated tags>"}.'
)


def _image_to_data_url(image_path, max_side=MAX_IMAGE_SIDE):
    """Image file -> data:image/jpeg;base64,...; raises on failure.

    Transparent images are composited onto white first (a direct RGB cast
    would give a black background), then scaled down to fit max_side.
    """
    try:
        max_side = int(max_side)
    except (TypeError, ValueError):
        max_side = MAX_IMAGE_SIDE
    max_side = max(256, max_side)
    with Image.open(image_path) as img:
        if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
            bg = Image.new("RGB", img.size, (255, 255, 255))
            bg.paste(img.convert("RGB"), mask=img.convert("RGBA").split()[-1])
            img = bg
        else:
            img = img.convert("RGB")
        w, h = img.size
        scale = min(1.0, max_side / max(w, h))
        if scale < 1.0:
            img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


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
                "image": (sorted(files), {"image_upload": True}),
                "instruction": ("STRING", {"multiline": True, "default": DEFAULT_INSTRUCTION}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.1, "max": 1.5, "step": 0.05}),
                "max_tokens": ("INT", {"default": 1024, "min": 64, "max": 8192, "step": 64}),
                "max_side": ("INT", {"default": 1024, "min": 256, "max": 4096, "step": 128}),
                "thinking_mode": (THINKING_MODES, {"default": THINKING_OURS}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("chinese", "english", "thinking")
    FUNCTION = "caption_image"
    CATEGORY = "FeiFei"

    def caption_image(self, image, instruction, temperature, max_tokens, max_side=1024, thinking_mode=THINKING_OURS):
        try:
            import folder_paths
            image_path = folder_paths.get_annotated_filepath(image)
        except Exception:
            image_path = image if isinstance(image, str) and os.path.isfile(image) else None
        if not image_path or not os.path.isfile(image_path):
            return (f"API Error: image file not found: {image}", "", "")

        api_base, api_key, model = resolve_endpoint("FeiFeiImageCaptioner")
        base = (api_base or "").strip().rstrip("/")
        if not base:
            return ("API Error: api_base is empty in config.json", "", "")

        try:
            data_url = _image_to_data_url(image_path, max_side)
        except Exception as e:
            return (f"API Error: failed to read/encode image: {e}", "", "")

        # Model native ignores the instruction box and uses the built-in minimal
        # instruction so the model's own thinking runs; Ours/Both use the box
        if thinking_mode == THINKING_MODEL:
            effective_instruction = NATIVE_INSTRUCTION
        else:
            effective_instruction = instruction or DEFAULT_INSTRUCTION
        enable_thinking = thinking_mode in (THINKING_MODEL, THINKING_BOTH)
        payload = {
            "messages": [
                {"role": "system", "content": "You are an expert image analyst. Always respond with valid JSON only."},
                {"role": "user", "content": [
                    {"type": "text", "text": effective_instruction},
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
            res_json = _post_chat_completions(base, payload, timeout=180, api_key=api_key)
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
            return (f"API Error: {e}", "", "")

        parsed = _extract_json_object(raw)
        if parsed is not None and (parsed.get("chinese") or parsed.get("english")):
            chinese = str(parsed.get("chinese", "") or "").strip()
            english = str(parsed.get("english", "") or "").strip()
            return (chinese, english, thinking)

        print("[FeiFeiImageCaptioner] model did not reply in JSON, full text goes to chinese.")
        return (raw.strip(), "", thinking)


NODE_CLASS_MAPPINGS = {"FeiFeiImageCaptioner": FeiFeiImageCaptioner}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FeiFeiImageCaptioner": "Image Captioner (API: edit config.json)",
}
