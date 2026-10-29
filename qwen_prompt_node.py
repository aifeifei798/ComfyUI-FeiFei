import os
import json
import re
import math
import urllib.request

from .llm_common import (
    SOURCE_API,
    SOURCE_LOCAL,
    SOURCE_MODES,
    THINKING_OURS,
    THINKING_MODEL,
    THINKING_BOTH,
    THINKING_MODES,
    _coerce_text,
    _fit_image_rgb,
    _pil_to_data_url,
    _post_chat_completions,
    _extract_json_object,
    _split_think_tags,
    _strip_v1,
    resolve_endpoint,
)
from .local_model import run_chat

# Recommended resolution per common ratio (based on ~1.5M/2K pixels, aligned to
# multiples of 16)
RATIO_MAP_2K = {
    "1:1": (1536, 1536),
    "3:2": (1872, 1248),
    "2:3": (1248, 1872),
    "16:9": (2016, 1152),
    "9:16": (1152, 2016),
    "4:3": (1728, 1296),
    "3:4": (1296, 1728),
    "21:9": (2304, 992),
    "2:1": (2048, 1024),
    "7:3": (2352, 1008),
    "18:39": (1056, 2288),
    "9:20": (1088, 2416),
    "5:7": (1280, 1792),
    "7:5": (1792, 1280),
}

DEFAULT_W, DEFAULT_H = 1536, 1536
# Clamp range for custom ratios; only affects ratios missing from RATIO_MAP_2K,
# so a bogus "100:1" cannot blow up the size
MIN_RATIO, MAX_RATIO = 0.25, 4.0

# Prompt families. Mode picks t2i/i2i; the Flux set additionally splits into a
# plain and a _bbox variant. Qwen replies with JSON carrying wh_ratio; Flux
# replies with the bare caption the image model consumes.
PROMPT_SET_QWEN = "Qwen-Image-2.1"
PROMPT_SET_FLUX = "Flux-3-Klein-PUP"
PROMPT_SETS = [PROMPT_SET_QWEN, PROMPT_SET_FLUX]
FLUX_PROMPT_FILE = "flux-3-image-klein-pup.system_prompts.json"

# Longest-side cap for reference images before upload, matching Image Captioner
MAX_IMAGE_SIDE = 1024

# Local mode has no max_tokens widget: the raw /completion fallback uses the same cap
LOCAL_MAX_TOKENS = 2048

# Minimal system prompt for the model-native chain: JSON shape only, no scaffolding
NATIVE_SYSTEM_PROMPT = (
    "You are an expert at enhancing image prompts for image generation. "
    "Think freely, then output ONLY one valid JSON object, no other text: "
    '{"rewritten_prompt": "<detailed English image description>", '
    '"wh_ratio": "<e.g. 3:2, empty string if unsure>", '
    '"ratio_follow": "<notes or empty string>"}.'
)

_CANVAS_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*[:：/]\s*(\d+(?:\.\d+)?)\s*$")


def parse_wh_ratio(ratio_str, target_pixel_count=1536*1536):
    """Compute width and height from an aspect ratio, aligned to multiples of 16"""
    if not isinstance(ratio_str, str):
        return DEFAULT_W, DEFAULT_H
    ratio_str = ratio_str.strip()
    if not ratio_str:
        return DEFAULT_W, DEFAULT_H
    if ratio_str in RATIO_MAP_2K:
        return RATIO_MAP_2K[ratio_str]

    # Parse a custom ratio such as "16:9"
    match = re.match(r"(\d+(?:\.\d+)?)\s*[:：/]\s*(\d+(?:\.\d+)?)", ratio_str)
    if match:
        w_factor = float(match.group(1))
        h_factor = float(match.group(2))
        if w_factor > 0 and h_factor > 0:
            ratio = w_factor / h_factor
            ratio = max(MIN_RATIO, min(MAX_RATIO, ratio))
            height = math.sqrt(target_pixel_count / ratio)
            width = height * ratio
            # Align to a multiple of 16
            width = int(round(width / 16.0) * 16)
            height = int(round(height / 16.0) * 16)
            if width <= 0 or height <= 0:
                return DEFAULT_W, DEFAULT_H
            return width, height

    # Default to 1:1
    return DEFAULT_W, DEFAULT_H


def resolve_canvas(canvas):
    """Validate the Flux "TARGET CANVAS" string -> (ratio, width, height).

    The ratio is normalized to "W:H"; empty or unparseable input falls back to
    ("", DEFAULT_W, DEFAULT_H). Pure function for unit tests.
    """
    text = (canvas or "").strip()
    if not text:
        return "", DEFAULT_W, DEFAULT_H
    if text in RATIO_MAP_2K:
        return text, *parse_wh_ratio(text)
    match = _CANVAS_RE.match(text)
    if not match:
        return "", DEFAULT_W, DEFAULT_H
    ratio = f"{float(match.group(1)):g}:{float(match.group(2)):g}"
    return ratio, *parse_wh_ratio(ratio)


def load_flux_system_prompt(mode, bbox=False):
    """Read one Flux prompt (t2i/i2i x plain/bbox) from the JSON next to this module.

    Returns "" when the file or the key is missing, so the caller can fall back.
    """
    key = ("t2i" if "T2I" in (mode or "") else "i2i") + ("_bbox" if bbox else "")
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), FLUX_PROMPT_FILE)
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError, UnicodeDecodeError) as e:
        print(f"[QwenImagePromptEnhancer] failed to read system prompts {path}: {e}")
        return ""
    text = data.get(key) if isinstance(data, dict) else None
    return text.strip() if isinstance(text, str) and text.strip() else ""


def _image_tensor_to_pil(image, max_side=MAX_IMAGE_SIDE):
    """IMAGE tensor -> RGB PIL (first frame of a batch, longest side capped); None on failure."""
    try:
        import numpy as np
        from PIL import Image

        frame = image[0] if isinstance(image, (list, tuple)) else image
        if frame is None:
            return None
        if hasattr(frame, "detach"):
            frame = frame.detach().cpu().float().numpy()
        if frame.ndim == 4:
            frame = frame[0]
        if frame.ndim == 2:
            frame = frame[..., None]
        if frame.shape[-1] == 4:
            frame = frame[..., :3]
        array = np.clip(frame * 255.0, 0, 255).astype("uint8")
        return _fit_image_rgb(Image.fromarray(array), max_side)
    except Exception as e:
        print(f"[QwenImagePromptEnhancer] failed to convert reference image: {e}")
        return None


def _image_to_data_url(image, max_side=MAX_IMAGE_SIDE):
    """IMAGE tensor -> data URL (first frame of a batch); returns "" on failure."""
    picture = _image_tensor_to_pil(image, max_side)
    return _pil_to_data_url(picture) if picture is not None else ""


def _user_content(text, image_urls):
    """User content: a plain string, or text plus image parts when references are attached."""
    if not image_urls:
        return text
    content = [{"type": "text", "text": text}]
    content.extend({"type": "image_url", "image_url": {"url": url}} for url in image_urls)
    return content


def _local_user_content(text, references):
    """Local messages: plain text, or text plus PIL parts when references are attached."""
    if not references:
        return text
    content = [{"type": "text", "text": text}]
    for image in references:
        picture = _image_tensor_to_pil(image)
        if picture is not None:
            content.append({"type": "image", "image": picture})
    return content


class QwenImagePromptEnhancer:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "source": (SOURCE_MODES, {
                    "default": SOURCE_API,
                    "tooltip": "API (config.json) sends the request to the endpoint in config.json; "
                               "Local HF model runs the `model_dir` folder listed in config.json with transformers.",
                }),
                "user_prompt": ("STRING", {"multiline": True, "default": "Tokyo Japanese girl walking in the rain with umbrella"}),
                "mode": (["T2I", "I2I"], {"default": "T2I"}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.1, "max": 1.5, "step": 0.05}),
                "thinking_mode": (THINKING_MODES, {"default": THINKING_OURS}),
                "prompt_set": (PROMPT_SETS, {
                    "default": PROMPT_SET_QWEN,
                    "tooltip": "Qwen-Image-2.1 replies with JSON and a wh_ratio; Flux-3-Klein-PUP replies with the bare caption (or instruction) the image model consumes.",
                }),
            },
            "optional": {
                "timeout": ("INT", {
                    "default": 120, "min": 10, "max": 600, "step": 5,
                    "tooltip": "API mode only: request timeout in seconds.",
                }),
                "bbox": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Flux set only: use the _bbox prompt (caption/instruction plus a JSON element table).",
                }),
                "canvas": ("STRING", {
                    "default": "",
                    "tooltip": "Flux T2I only: target canvas as W:H, sent as a TARGET CANVAS line and echoed to wh_ratio / width / height. Leave empty to let the model compose freely.",
                }),
                "image_1": ("IMAGE",),
                "image_2": ("IMAGE",),
                "image_3": ("IMAGE",),
                "image_4": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "INT", "INT", "STRING", "STRING")
    RETURN_NAMES = ("rewritten_prompt", "wh_ratio", "width", "height", "ratio_follow", "thinking")
    FUNCTION = "enhance_prompt"
    CATEGORY = "FeiFei"

    def load_system_prompt(self, mode, thinking_mode, prompt_set, bbox):
        if thinking_mode == THINKING_MODEL:
            return NATIVE_SYSTEM_PROMPT
        if prompt_set == PROMPT_SET_FLUX:
            content = load_flux_system_prompt(mode, bbox)
            if content:
                return content
            print(f"[QwenImagePromptEnhancer] no Flux prompt for {mode} (bbox={bbox}), using the Qwen file")
        current_dir = os.path.dirname(os.path.abspath(__file__))
        if "T2I" in mode:
            filename = "Qwen-Image-2.1-T2I.system_prompt.txt"
        else:
            filename = "Qwen-Image-2.1-I2I.system_prompt.txt"

        filepath = os.path.join(current_dir, filename)
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()
                if content.strip():
                    return content
        except (OSError, UnicodeDecodeError) as e:
            print(f"[QwenImagePromptEnhancer] failed to read system prompt {filepath}: {e}")
        return "You are an expert at enhancing image prompts. Output valid JSON."

    def enhance_prompt(self, source, user_prompt, mode, temperature,
                       thinking_mode=THINKING_OURS, prompt_set=PROMPT_SET_QWEN, timeout=120,
                       bbox=False, canvas="", image_1=None, image_2=None, image_3=None, image_4=None):
        # The Flux prompts are the caption itself; "Model native" still asks for
        # our JSON shape, so the two behaviours stay separate.
        use_flux_prompt = prompt_set == PROMPT_SET_FLUX and thinking_mode in (THINKING_OURS, THINKING_BOTH)
        system_prompt = self.load_system_prompt(mode, thinking_mode, prompt_set, bbox)

        # The Flux prompts compose for a canvas handed to them; the Qwen prompts
        # choose the ratio themselves, so canvas only applies to Flux T2I.
        request_text = user_prompt
        if use_flux_prompt and "T2I" in mode:
            canvas_text, _, _ = resolve_canvas(canvas)
            if canvas_text:
                request_text = f"TARGET CANVAS: {canvas_text}\n{user_prompt}"

        # Reference images ground an editing request; the Qwen I2I file also
        # acts on the image it sees
        references = []
        if "I2I" in mode:
            references = [image for image in (image_1, image_2, image_3, image_4) if image is not None]

        # Ours runs only our prompt chain (model-native thinking off); Model/Both turn it on
        enable_thinking = thinking_mode in (THINKING_MODEL, THINKING_BOTH)

        if source == SOURCE_LOCAL:
            raw_content, thinking, error = self._run_local(
                system_prompt, request_text, references, temperature, enable_thinking)
        else:
            raw_content, thinking, error = self._run_api(
                system_prompt, request_text, references, temperature, timeout, enable_thinking, use_flux_prompt)
        if error:
            return (error, "", DEFAULT_W, DEFAULT_H, "", "")

        # The Flux reply is already the text the image model receives
        if use_flux_prompt:
            if "T2I" in mode:
                wh_ratio, width, height = resolve_canvas(canvas)
            else:
                wh_ratio, width, height = "", DEFAULT_W, DEFAULT_H
            return (raw_content.strip(), wh_ratio, width, height, "", thinking)

        # Parse the LLM's JSON
        rewritten_prompt = raw_content
        wh_ratio = "3:2" if "T2I" in mode else ""
        ratio_follow = ""

        parsed = _extract_json_object(raw_content)
        if parsed is not None:
            rewritten_prompt = parsed.get("rewritten_prompt", raw_content) or raw_content
            if isinstance(parsed.get("wh_ratio"), str):
                wh_ratio = parsed.get("wh_ratio", "").strip()
            if isinstance(parsed.get("ratio_follow"), str):
                ratio_follow = parsed.get("ratio_follow", "")

        # Compute the size (I2I returns an empty string meaning "keep the source
        # size", so the defaults are just a placeholder there)
        width, height = parse_wh_ratio(wh_ratio if wh_ratio else "1:1")

        return (rewritten_prompt, wh_ratio, width, height, ratio_follow, thinking)

    def _run_local(self, system_prompt, request_text, references, temperature, enable_thinking):
        """Run the local HF folder; returns (content, thinking, error)."""
        return run_chat("QwenImagePromptEnhancer", system_prompt,
                        _local_user_content(request_text, references), temperature, enable_thinking)

    def _run_api(self, system_prompt, request_text, references, temperature, timeout,
                 enable_thinking, use_flux_prompt):
        """POST the OpenAI-compatible request; returns (content, thinking, error)."""
        api_base, api_key, model = resolve_endpoint("QwenImagePromptEnhancer")
        base = (api_base or "").strip().rstrip("/")
        if not base:
            return "", "", "API Error: api_base is empty in config.json"

        image_urls = [url for url in (_image_to_data_url(image) for image in references) if url]

        model_name = (model or "").strip()
        # Check whether the current model is a Gemma variant
        is_gemma = "gemma" in model_name.lower()

        # 1. Work around Gemma rejecting the system role
        if is_gemma:
            # Gemma's own convention: merge system instructions to the front of user
            text = f"[System Instructions]\n{system_prompt}\n\n[User Request]\n{request_text}"
            messages = [{"role": "user", "content": _user_content(text, image_urls)}]
        else:
            # Standard two-role layout for Qwen / LLaMA and friends
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": _user_content(request_text, image_urls)},
            ]

        # 2. Build a standard OpenAI-compatible payload
        payload = {
            "messages": messages,
            "temperature": temperature,
            "stream": False,
            "enable_thinking": enable_thinking,
        }
        # The Flux prompts answer with the caption itself (plus a bare JSON array
        # for the bbox variants), so force JSON only for the Qwen / native prompts
        if not use_flux_prompt:
            payload["response_format"] = {"type": "json_object"}
        if model_name:
            payload["model"] = model_name

        raw_content = ""
        thinking = ""
        first_error = ""
        try:
            seconds = int(timeout)
        except (TypeError, ValueError):
            seconds = 120
        seconds = max(10, min(600, seconds))
        try:
            res_json = _post_chat_completions(base, payload, timeout=seconds, api_key=api_key)
            # OpenAI-compatible shape: choices[0].message.content + reasoning_content
            choices = res_json.get("choices") if isinstance(res_json, dict) else None
            if choices:
                message = choices[0].get("message", {}) if isinstance(choices[0], dict) else {}
                raw_content = _coerce_text(message.get("content", ""))
                thinking = _coerce_text(message.get("reasoning_content", ""))
            if not raw_content:
                raise ValueError("LLM response missing choices[0].message.content")
        except Exception as e:
            first_error = str(e)
            # llama.cpp's /completion has no image support, so a request carrying
            # reference images cannot fall back to it
            if image_urls:
                return "", "", f"API Error: {first_error}"
            # Fall back to llama.cpp's native /completion endpoint
            raw_url = _strip_v1(base) + "/completion"

            # The fallback endpoint gets the same Gemma / Qwen templating
            if is_gemma:
                raw_prompt_text = (
                    f"<start_of_turn>user\n"
                    f"{system_prompt}\n\n{request_text}<end_of_turn>\n"
                    f"<start_of_turn>model\n"
                )
            else:
                raw_prompt_text = (
                    f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
                    f"<|im_start|>user\n{request_text}<|im_end|>\n"
                    f"<|im_start|>assistant\n"
                )

            raw_payload = {
                "prompt": raw_prompt_text,
                "temperature": temperature,
                "n_predict": 2048
            }
            try:
                req_raw = urllib.request.Request(
                    raw_url,
                    data=json.dumps(raw_payload).encode("utf-8"),
                    headers={"Content-Type": "application/json"}
                )
                with urllib.request.urlopen(req_raw, timeout=seconds) as resp:
                    resp_body = json.loads(resp.read().decode("utf-8", errors="replace"))
                    raw_content = (resp_body.get("content") or "").strip()
                    thinking, raw_content = _split_think_tags(raw_content)
                    if not raw_content:
                        raise ValueError(f"/completion response missing content: {str(resp_body)[:500]}")
            except Exception as ex:
                return "", "", f"API Error: {first_error} / {str(ex)}"

        return raw_content, thinking, ""


NODE_CLASS_MAPPINGS = {"QwenImagePromptEnhancer": QwenImagePromptEnhancer}
NODE_DISPLAY_NAME_MAPPINGS = {
    "QwenImagePromptEnhancer": "Prompt Enhancer (API / Local HF)",
}
