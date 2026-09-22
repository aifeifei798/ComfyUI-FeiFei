"""图生制作词节点：上传图片，经视觉语言模型（OpenAI 兼容接口）生成中英制作词。

要求 api_base 背后是视觉模型（如 Qwen-VL 系 / MiniCPM-V 等），llama.cpp
原生 /completion 接口不支持图片，本节点只走 /v1/chat/completions。
"""

import base64
import io
import json
import os
import urllib.request

from PIL import Image

from .qwen_prompt_node import _extract_json_object  # 复用括号配平 JSON 提取

DEFAULT_INSTRUCTION = (
    "Look at this image carefully and output ONLY one JSON object, nothing else: "
    '{"chinese": "describe the image content, subject, action, scene, lighting and style in detail in Chinese", '
    '"english": "English image-generation prompt, comma-separated tags and quality words, '
    "directly usable in Stable Diffusion or Qwen-Image, e.g. '1girl, ... , masterpiece, best quality'\"}"
)

# 传图前最长边压缩上限（控制 base64 体积）
MAX_IMAGE_SIDE = 1024


def _image_to_data_url(image_path):
    """图片文件 -> data:image/jpeg;base64,...；失败抛异常"""
    with Image.open(image_path) as img:
        img = img.convert("RGB")
        w, h = img.size
        scale = min(1.0, MAX_IMAGE_SIDE / max(w, h))
        if scale < 1.0:
            img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


def _content_to_text(content):
    """兼容 content 为字符串或 parts 列表两种回包"""
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
                "api_base": ("STRING", {"default": "http://127.0.0.1:8080"}),
                "model": ("STRING", {"multiline": False, "default": ""}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.1, "max": 1.5, "step": 0.05}),
                "max_tokens": ("INT", {"default": 1024, "min": 64, "max": 8192, "step": 64}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("chinese", "english")
    FUNCTION = "caption_image"
    CATEGORY = "FeiFei"

    def caption_image(self, image, instruction, api_base, model, temperature, max_tokens):
        try:
            import folder_paths
            image_path = folder_paths.get_annotated_filepath(image)
        except Exception:
            image_path = image if isinstance(image, str) and os.path.isfile(image) else None
        if not image_path or not os.path.isfile(image_path):
            return (f"API Error: image file not found: {image}", "")

        base = (api_base or "").strip().rstrip("/")
        if not base:
            return ("API Error: api_base is empty", "")

        try:
            data_url = _image_to_data_url(image_path)
        except Exception as e:
            return (f"API Error: failed to read/encode image: {e}", "")

        payload = {
            "messages": [
                {"role": "system", "content": "You are an expert image analyst. Always respond with valid JSON only."},
                {"role": "user", "content": [
                    {"type": "text", "text": instruction or DEFAULT_INSTRUCTION},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ]},
            ],
            "temperature": temperature,
            "max_tokens": int(max_tokens),
            "stream": False,
        }
        model_name = (model or "").strip()
        if model_name:
            payload["model"] = model_name

        try:
            req = urllib.request.Request(
                base + "/v1/chat/completions",
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=180) as response:
                body = response.read().decode("utf-8", errors="replace")
                res_json = json.loads(body)
                choices = res_json.get("choices") if isinstance(res_json, dict) else None
                raw = ""
                if choices and isinstance(choices[0], dict):
                    raw = _content_to_text(choices[0].get("message", {}).get("content", ""))
                if not raw:
                    raise ValueError(f"LLM response missing text content: {body[:500]}")
        except Exception as e:
            return (f"API Error: {e}", "")

        parsed = _extract_json_object(raw)
        if parsed is not None and (parsed.get("chinese") or parsed.get("english")):
            chinese = str(parsed.get("chinese", "") or "").strip()
            english = str(parsed.get("english", "") or "").strip()
            return (chinese, english)

        print("[FeiFeiImageCaptioner] model did not reply in JSON, full text goes to chinese.")
        return (raw.strip(), "")


NODE_CLASS_MAPPINGS = {"FeiFeiImageCaptioner": FeiFeiImageCaptioner}

NODE_DISPLAY_NAME_MAPPINGS = {"FeiFeiImageCaptioner": "Image Captioner"}
