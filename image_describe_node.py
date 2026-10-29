"""Image describe node: upload one image, let a vision model describe it, and
return that text as a single string.

Deliberately plain: no JSON, no language switch, no extra outputs. Whatever the
model writes is the answer, minus any <think>...</think> reasoning block it
leaves in the reply, so the output can be wired straight into a prompt box.

Two sources, picked with `source`:

- API (config.json): POST /v1/chat/completions to the OpenAI-compatible endpoint
  in config.json, which must serve a vision model. llama.cpp's native /completion
  endpoint has no image support, so this path only uses /v1/chat/completions.
- Local HF model: load an HF-format folder with transformers, in-process. Nothing
  is ever downloaded here: a folder without a config and weights returns the
  `hf download` command instead of fetching anything.
"""

import gc
import os
import threading

from .llm_common import (
    _coerce_text,
    _image_file_to_data_url,
    _open_image_rgb,
    _post_chat_completions,
    _strip_thinking,
    resolve_endpoint,
)

# Longest-side cap before the image leaves the node, keeps payloads and tokens small
MAX_IMAGE_SIDE = 1024

SOURCE_API = "API (config.json)"
SOURCE_LOCAL = "Local HF model"
SOURCE_MODES = [SOURCE_API, SOURCE_LOCAL]

DEFAULT_INSTRUCTION = (
    "Describe this image in detail: subject, action, scene, lighting and style. "
    "Plain prose in one paragraph, no lists, no JSON, no preamble."
)

SYSTEM_PROMPT = (
    "You are an expert image analyst. Answer with the description itself only; "
    "no reasoning, no summary of the task, no alternatives."
)

# A folder counts as a usable local model with a config plus one weight file
WEIGHT_SUFFIXES = (".safetensors", ".bin", ".pt", ".pth")
REQUIRED_TRANSFORMERS = "5.21"

_LOCAL_LOCK = threading.Lock()
_LOCAL_MODELS = {}


def resolve_model_dir(model_dir):
    """Widget value -> absolute path; an empty or blank widget stays empty."""
    text = (model_dir or "").strip()
    if not text:
        return ""
    return os.path.abspath(os.path.expanduser(text))


def model_files_ready(model_dir):
    """True when the folder exists and holds a config.json plus a weight file."""
    if not model_dir or not os.path.isdir(model_dir):
        return False
    try:
        entries = os.listdir(model_dir)
    except OSError:
        return False
    names = [name.lower() for name in entries]
    return "config.json" in names and any(n.endswith(WEIGHT_SUFFIXES) for n in names)


def build_download_command(hf_model, model_dir):
    """The command the user runs themselves; the node never downloads anything."""
    repo = (hf_model or "").strip() or "<hf-repo-id>"
    target = (model_dir or "").strip() or "<local-dir>"
    return f'hf download {repo} --local-dir "{target}"'


def missing_model_hint(hf_model, model_dir):
    where = (model_dir or "").strip() or "(model_dir is empty)"
    return (
        f"Local model not found: {where}\n"
        f"Expected config.json plus a weight file ({', '.join(WEIGHT_SUFFIXES)}) in that folder.\n"
        "Download it, then run this node again:\n"
        f"  {build_download_command(hf_model, model_dir)}\n"
        f"Loading it needs transformers>={REQUIRED_TRANSFORMERS} in the ComfyUI environment."
    )


def _release_local_models(keep=None):
    """Drop every cached model except `keep`, then hand the VRAM back."""
    dropped = [key for key in _LOCAL_MODELS if key != keep]
    for key in dropped:
        _LOCAL_MODELS.pop(key, None)
    if dropped:
        print(f"[FeiFeiImageDescribe] releasing local model: {', '.join(dropped)}")
        gc.collect()
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass


def _load_local_model(model_dir):
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    if torch.cuda.is_available():
        device = "cuda"
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    else:
        device = "cpu"
        dtype = torch.float32
    print(f"[FeiFeiImageDescribe] loading {model_dir} on {device} ({dtype}) ...")
    model = AutoModelForImageTextToText.from_pretrained(
        model_dir,
        dtype=dtype,
        trust_remote_code=True,
    ).eval().to(device)
    processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True)
    return model, processor


def _get_local_model(model_dir):
    with _LOCAL_LOCK:
        _release_local_models(keep=model_dir)
        loaded = _LOCAL_MODELS.get(model_dir)
        if loaded is None:
            loaded = _load_local_model(model_dir)
            _LOCAL_MODELS[model_dir] = loaded
        return loaded


def describe_local_model(model_dir, image_path, instruction, temperature, max_tokens, max_side):
    """Run the local folder and return the reply text; raises on any failure."""
    import torch

    model, processor = _get_local_model(model_dir)
    image = _open_image_rgb(image_path, max_side)
    messages = [{"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": instruction or DEFAULT_INSTRUCTION},
    ]}]
    # Chat templates read enable_thinking; the ones that do not simply ignore it
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        enable_thinking=False,
    )
    inputs = {k: (v.to(model.device) if torch.is_tensor(v) else v) for k, v in inputs.items()}

    kwargs = {"max_new_tokens": max(16, int(max_tokens)), "repetition_penalty": 1.1}
    if temperature and temperature > 0:
        kwargs.update(do_sample=True, temperature=float(temperature), top_k=50)
    else:
        kwargs["do_sample"] = False

    generated = model.generate(**inputs, **kwargs)
    prompt_len = inputs["input_ids"].shape[-1]
    return _strip_thinking(processor.tokenizer.decode(generated[0][prompt_len:], skip_special_tokens=True))


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
                               "Local HF model loads the folder in model_dir with transformers.",
                }),
                "hf_model": ("STRING", {
                    "default": "LiquidAI/LFM3-VL-2B",
                    "tooltip": "Hugging Face repo id, used only for the download command shown when "
                               "model_dir has no model yet.",
                }),
                "model_dir": ("STRING", {
                    "default": "",
                    "tooltip": "Local HF-format model folder (config.json + weights). Absolute path, "
                               "or relative to the ComfyUI working directory.",
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

    def describe_image(self, source, hf_model, model_dir, image, instruction, temperature, max_tokens,
                       max_side=MAX_IMAGE_SIDE, timeout=180):
        try:
            import folder_paths
            image_path = folder_paths.get_annotated_filepath(image)
        except Exception:
            image_path = image if isinstance(image, str) and os.path.isfile(image) else None
        if not image_path or not os.path.isfile(image_path):
            return (f"Error: image file not found: {image}",)

        if source == SOURCE_LOCAL:
            return (self._describe_local(hf_model, model_dir, image_path,
                                         instruction, temperature, max_tokens, max_side),)
        return (self._describe_api(image_path, instruction, temperature, max_tokens, max_side, timeout),)

    def _describe_local(self, hf_model, model_dir, image_path, instruction, temperature, max_tokens, max_side):
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
