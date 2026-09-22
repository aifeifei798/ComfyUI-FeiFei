import os
import json
import re
import math
import urllib.request
import urllib.error

from .llm_common import (
    THINKING_OURS,
    THINKING_MODEL,
    THINKING_BOTH,
    THINKING_MODES,
    _coerce_text,
    _post_chat_completions,
    _strip_code_fences,
    _extract_json_object,
)

# 常见比例对应的推荐分辨率 (以 ~1.5M/2K 像素为基准，对齐 16 的倍数)
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

# 模型原生链用的极简 system prompt：只规定 JSON 格式，不给推导脚手架
NATIVE_SYSTEM_PROMPT = (
    "You are an expert at enhancing image prompts for image generation. "
    "Think freely, then output ONLY one valid JSON object, no other text: "
    '{"rewritten_prompt": "<detailed English image description>", '
    '"wh_ratio": "<e.g. 3:2, empty string if unsure>", '
    '"ratio_follow": "<notes or empty string>"}.'
)

def _split_think_tags(text):
    """从 <think>...</think> 内联标签拆出 (thinking, content)，无标签返回 ("", text)"""
    if not isinstance(text, str) or "<think>" not in text.lower():
        return "", text if isinstance(text, str) else ""
    match = re.search(r"<think>(.*?)</think>", text, re.DOTALL | re.IGNORECASE)
    if not match:
        return "", text
    thinking = match.group(1).strip()
    content = (text[:match.start()] + text[match.end():]).strip()
    return thinking, content


def parse_wh_ratio(ratio_str, target_pixel_count=1536*1536):
    """根据宽高比计算宽和高（16的倍数）"""
    if not isinstance(ratio_str, str):
        return DEFAULT_W, DEFAULT_H
    ratio_str = ratio_str.strip()
    if not ratio_str:
        return DEFAULT_W, DEFAULT_H
    if ratio_str in RATIO_MAP_2K:
        return RATIO_MAP_2K[ratio_str]
    
    # 解析自定义比例如 "16:9"
    match = re.match(r"(\d+(?:\.\d+)?)\s*[:：/]\s*(\d+(?:\.\d+)?)", ratio_str)
    if match:
        w_factor = float(match.group(1))
        h_factor = float(match.group(2))
        if w_factor > 0 and h_factor > 0:
            ratio = w_factor / h_factor
            height = math.sqrt(target_pixel_count / ratio)
            width = height * ratio
            # 对齐到 16 的倍数
            width = int(round(width / 16.0) * 16)
            height = int(round(height / 16.0) * 16)
            if width <= 0 or height <= 0:
                return DEFAULT_W, DEFAULT_H
            return width, height

    # 默认 1:1
    return DEFAULT_W, DEFAULT_H

class QwenImagePromptEnhancer:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "user_prompt": ("STRING", {"multiline": True, "default": "Tokyo Japanese girl walking in the rain with umbrella"}),
                "mode": (["T2I", "I2I"], {"default": "T2I"}),
                "api_base": ("STRING", {"default": "http://127.0.0.1:8080"}),
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

    def enhance_prompt(self, user_prompt, mode, api_base, temperature, thinking_mode=THINKING_OURS):
        system_prompt = self.load_system_prompt(mode, thinking_mode)
        base = (api_base or "").strip().rstrip("/")
        if not base:
            return ("API Error: api_base is empty", "", DEFAULT_W, DEFAULT_H, "", "")

        # Ours 只走咱们定义的 8 步链（关模型原生思考）；Model/Both 打开模型自带思考
        enable_thinking = thinking_mode in (THINKING_MODEL, THINKING_BOTH)
        payload = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": temperature,
            "stream": False,
            "enable_thinking": enable_thinking,
        }

        raw_content = ""
        thinking = ""
        first_error = ""
        try:
            res_json = _post_chat_completions(base, payload, timeout=120)
            # OpenAI 兼容格式：choices[0].message.content + reasoning_content（思考过程）
            choices = res_json.get("choices") if isinstance(res_json, dict) else None
            if choices:
                message = choices[0].get("message", {}) if isinstance(choices[0], dict) else {}
                raw_content = _coerce_text(message.get("content", ""))
                thinking = _coerce_text(message.get("reasoning_content", ""))
            if not raw_content:
                raise ValueError("LLM response missing choices[0].message.content")
        except Exception as e:
            first_error = str(e)
            # 兼容 llama.cpp 原生 /completion 接口（思考以内联 <think> 标签返回）
            raw_url = base + "/completion"
            raw_payload = {
                "prompt": f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n",
                "temperature": temperature,
                "n_predict": 2048
            }
            try:
                req_raw = urllib.request.Request(raw_url, data=json.dumps(raw_payload).encode("utf-8"), headers={"Content-Type": "application/json"})
                with urllib.request.urlopen(req_raw, timeout=120) as resp:
                    resp_body = json.loads(resp.read().decode("utf-8", errors="replace"))
                    raw_content = (resp_body.get("content") or "").strip()
                    thinking, raw_content = _split_think_tags(raw_content)
                    if not raw_content:
                        raise ValueError(f"/completion response missing content: {str(resp_body)[:500]}")
            except Exception as ex:
                return (f"API Error: {first_error} / {str(ex)}", "", DEFAULT_W, DEFAULT_H, "", "")

        # 解析 LLM 返回的 JSON
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

        # 计算尺寸（I2I 返回空字符串表示“保持原图”，尺寸给默认值占位）
        width, height = parse_wh_ratio(wh_ratio if wh_ratio else "1:1")

        return (rewritten_prompt, wh_ratio, width, height, ratio_follow, thinking)