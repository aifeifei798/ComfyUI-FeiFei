import os
import json
import re
import math
import urllib.request

from .llm_common import (
    THINKING_OURS,
    THINKING_MODEL,
    THINKING_BOTH,
    THINKING_MODES,
    _coerce_text,
    _post_chat_completions,
    _extract_json_object,
    _strip_v1,
)

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

# Minimal system prompt for the model-native chain: JSON shape only, no scaffolding
NATIVE_SYSTEM_PROMPT = (
    "You are an expert at enhancing image prompts for image generation. "
    "Think freely, then output ONLY one valid JSON object, no other text: "
    '{"rewritten_prompt": "<detailed English image description>", '
    '"wh_ratio": "<e.g. 3:2, empty string if unsure>", '
    '"ratio_follow": "<notes or empty string>"}.'
)

def _split_think_tags(text):
    """Split inline <think>...</think> tags into (thinking, content); ("", text) when absent"""
    if not isinstance(text, str) or "<think>" not in text.lower():
        return "", text if isinstance(text, str) else ""
    match = re.search(r"<think>(.*?)</think>", text, re.DOTALL | re.IGNORECASE)
    if not match:
        return "", text
    thinking = match.group(1).strip()
    content = (text[:match.start()] + text[match.end():]).strip()
    return thinking, content


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

class QwenImagePromptEnhancer:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "user_prompt": ("STRING", {"multiline": True, "default": "Tokyo Japanese girl walking in the rain with umbrella"}),
                "mode": (["T2I", "I2I"], {"default": "T2I"}),
                "api_base": ("STRING", {"default": "http://127.0.0.1:8080"}),
                "api_key": (
                    "STRING",
                    {"default": "", "multiline": False},
                ),
                "model": ("STRING", {"multiline": False, "default": ""}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.1, "max": 1.5, "step": 0.05}),
                "thinking_mode": (THINKING_MODES, {"default": THINKING_OURS}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "INT", "INT", "STRING", "STRING")
    RETURN_NAMES = ("rewritten_prompt", "wh_ratio", "width", "height", "ratio_follow", "thinking")
    FUNCTION = "enhance_prompt"
    CATEGORY = "FeiFei"

    def load_system_prompt(self, mode, thinking_mode=THINKING_OURS):
        if thinking_mode == THINKING_MODEL:
            return NATIVE_SYSTEM_PROMPT
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

    def enhance_prompt(self, user_prompt, mode, api_base, api_key, model, temperature, thinking_mode=THINKING_OURS):
        system_prompt = self.load_system_prompt(mode, thinking_mode)
        base = (api_base or "").strip().rstrip("/")
        if not base:
            return ("API Error: api_base is empty", "", DEFAULT_W, DEFAULT_H, "", "")

        model_name = (model or "").strip()
        # Check whether the current model is a Gemma variant
        is_gemma = "gemma" in model_name.lower()

        # 1. Work around Gemma rejecting the system role
        if is_gemma:
            # Gemma's own convention: merge system instructions to the front of user
            messages = [
                {
                    "role": "user",
                    "content": f"[System Instructions]\n{system_prompt}\n\n[User Request]\n{user_prompt}"
                }
            ]
        else:
            # Standard two-role layout for Qwen / LLaMA and friends
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]

        # Ours runs only our 8-step chain (model-native thinking off); Model/Both turn it on
        enable_thinking = thinking_mode in (THINKING_MODEL, THINKING_BOTH)
        
        # 2. Build a standard OpenAI-compatible payload
        payload = {
            "messages": messages,
            "temperature": temperature,
            "stream": False,
            "enable_thinking": enable_thinking,
            # Force pure JSON output so the model emits nothing but the object
            "response_format": {"type": "json_object"},
        }
        if model_name:
            payload["model"] = model_name

        raw_content = ""
        thinking = ""
        first_error = ""
        try:
            res_json = _post_chat_completions(base, payload, timeout=120, api_key=api_key)
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
            # Fall back to llama.cpp's native /completion endpoint
            raw_url = _strip_v1(base) + "/completion"
            
            # The fallback endpoint gets the same Gemma / Qwen templating
            if is_gemma:
                raw_prompt_text = (
                    f"<start_of_turn>user\n"
                    f"{system_prompt}\n\n{user_prompt}<end_of_turn>\n"
                    f"<start_of_turn>model\n"
                )
            else:
                raw_prompt_text = (
                    f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
                    f"<|im_start|>user\n{user_prompt}<|im_end|>\n"
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
                with urllib.request.urlopen(req_raw, timeout=120) as resp:
                    resp_body = json.loads(resp.read().decode("utf-8", errors="replace"))
                    raw_content = (resp_body.get("content") or "").strip()
                    thinking, raw_content = _split_think_tags(raw_content)
                    if not raw_content:
                        raise ValueError(f"/completion response missing content: {str(resp_body)[:500]}")
            except Exception as ex:
                return (f"API Error: {first_error} / {str(ex)}", "", DEFAULT_W, DEFAULT_H, "", "")

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


NODE_CLASS_MAPPINGS = {"QwenImagePromptEnhancer": QwenImagePromptEnhancer}
NODE_DISPLAY_NAME_MAPPINGS = {"QwenImagePromptEnhancer": "Qwen-Image Prompt Enhancer (LLaMA)"}