"""Image describe node: upload one image, let a vision model describe it, and
return that text as a single string.

Deliberately plain: no JSON, no language switch, no extra outputs. Whatever the
model writes is the answer, minus any <think>...</think> reasoning block it
leaves in the reply, so the output can be wired straight into a prompt box.

Two sources, picked with `source`:

- API (config.json): POST /v1/chat/completions to the OpenAI-compatible endpoint
  in config.json, which must serve a vision model. llama.cpp's native /completion
  endpoint has no image support, so this path only uses /v1/chat/completions.
- Local HF model: run the HF-format folder in `model_dir` in-process through
  transformers. A folder without a config and weights returns the `hf download`
  command instead of fetching anything.
"""

import os

from .llm_common import (
    SOURCE_API,
    SOURCE_LOCAL,
    SOURCE_MODES,
    _coerce_text,
    _image_file_to_data_url,
    _open_image_rgb,
    _post_chat_completions,
    _strip_thinking,
    resolve_endpoint,
    resolve_local_model,
)
from .local_model import (
    REQUIRED_TRANSFORMERS,
    generate,
    missing_model_hint,
    model_files_ready,
    resolve_model_dir,
)

# Longest-side cap before the image leaves the node, keeps payloads and tokens small
MAX_IMAGE_SIDE = 1024

DEFAULT_INSTRUCTION = (
    "Describe this image in detail: subject, action, scene, lighting and style. "
    "Plain prose in one paragraph, no lists, no JSON, no preamble."
)

SYSTEM_PROMPT = (
    "You are an expert image analyst. Answer with the description itself only; "
    "no reasoning, no summary of the task, no alternatives."
)


def describe_local_model(model_dir, image_path, instruction, temperature, max_tokens, max_side):
    """Run the local folder and return the reply text; raises on any failure."""
    messages = [{"role": "user", "content": [
        {"type": "image", "image": _open_image_rgb(image_path, max_side)},
        {"type": "text", "text": instruction or DEFAULT_INSTRUCTION},
    ]}]
    return generate(model_dir, messages, temperature=temperature, max_tokens=max_tokens)


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
                "source": (SOURCE_MODES, {
                    "default": SOURCE_API,
                    "tooltip": "API (config.json) posts the image to the endpoint in config.json; "
                               "Local HF model runs the `model_dir` folder listed in config.json with transformers.",
                }),
                "image": (sorted(files), {"image_upload": True}),
                "instruction": ("STRING", {"multiline": True, "default": DEFAULT_INSTRUCTION}),
                "temperature": ("FLOAT", {"default": 0.3, "min": 0.1, "max": 1.5, "step": 0.05}),
                "max_tokens": ("INT", {"default": 1024, "min": 64, "max": 8192, "step": 64}),
                "max_side": ("INT", {"default": MAX_IMAGE_SIDE, "min": 256, "max": 4096, "step": 128}),
            },
            "optional": {
                "timeout": ("INT", {
                    "default": 180, "min": 10, "max": 600, "step": 5,
                    "tooltip": "API mode only: request timeout in seconds.",
                }),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("description",)
    FUNCTION = "describe_image"
    CATEGORY = "FeiFei"

    def describe_image(self, source, image, instruction, temperature, max_tokens,
                       max_side=MAX_IMAGE_SIDE, timeout=180):
        try:
            import folder_paths
            image_path = folder_paths.get_annotated_filepath(image)
        except Exception:
            image_path = image if isinstance(image, str) and os.path.isfile(image) else None
        if not image_path or not os.path.isfile(image_path):
            return (f"Error: image file not found: {image}",)

        if source == SOURCE_LOCAL:
            return (self._describe_local(image_path, instruction, temperature, max_tokens, max_side),)
        return (self._describe_api(image_path, instruction, temperature, max_tokens, max_side, timeout),)

    def _describe_local(self, image_path, instruction, temperature, max_tokens, max_side):
        hf_model, model_dir = resolve_local_model("FeiFeiImageDescribe")
        folder = resolve_model_dir(model_dir)
        if not model_files_ready(folder):
            hint = missing_model_hint(hf_model, folder)
            print(f"[FeiFeiImageDescribe] {hint}")
            return hint
        try:
            text = describe_local_model(folder, image_path, instruction, temperature, max_tokens, max_side)
        except ImportError as e:
            return (f'Local Error: {e}; install it with: pip install -U "transformers>={REQUIRED_TRANSFORMERS}"')
        except Exception as e:
            return f"Local Error: {e}"
        text = _strip_thinking(text)
        if not text:
            return "Local Error: model returned an empty reply"
        return text

    def _describe_api(self, image_path, instruction, temperature, max_tokens, max_side, timeout):
        api_base, api_key, model = resolve_endpoint("FeiFeiImageDescribe")
        base = (api_base or "").strip().rstrip("/")
        if not base:
            return "API Error: api_base is empty in config.json"

        try:
            data_url = _image_file_to_data_url(image_path, max_side)
        except Exception as e:
            return f"API Error: failed to read/encode image: {e}"

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
            return f"API Error: {e}"

        if not text:
            return "API Error: model returned an empty reply"
        return text


NODE_CLASS_MAPPINGS = {"FeiFeiImageDescribe": FeiFeiImageDescribe}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FeiFeiImageDescribe": "Image Describe (API / Local HF)",
}
