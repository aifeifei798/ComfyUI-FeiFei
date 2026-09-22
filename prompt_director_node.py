"""Prompt Director：极简关键词 → MiMo Flash 一次吐结构化导演 JSON。

输出适合目标出图模型的正向/反向 prompt、镜头光影、宽高比
（钳制在 aspect_ratio_node.ASPECT_RATIOS 内，可直连 Aspect Ratio 节点）。
LLM 失败时返回错误字符串，不抛异常炸工作流。
"""

import re

from .llm_common import (
    THINKING_OURS,
    THINKING_MODEL,
    THINKING_BOTH,
    THINKING_MODES,
    _coerce_text,
    _post_chat_completions,
    _extract_json_object,
)
from .aspect_ratio_node import ASPECT_RATIOS

DEFAULT_RATIO = "1:1"

MODEL_STYLES = ["Flux", "SDXL", "Qwen-Image"]

# 各出图模型的正向词写法规则（注入 system prompt）
STYLE_RULES = {
    "Flux": (
        "Flux: write the positive prompt as flowing natural-language prose "
        "(English preferred), full descriptive sentences separated by commas. "
        "Do NOT use booru-style tag spam or quality-word stacks; bake detail "
        "into the sentences themselves."
    ),
    "SDXL": (
        "SDXL: write the positive prompt as comma-separated short tags in "
        "Danbooru style, starting with subject tags, then composition, then "
        "lighting/style tags, ending with quality tags like "
        "'masterpiece, best quality, highres, extremely detailed'."
    ),
    "Qwen-Image": (
        "Qwen-Image: write the positive prompt as natural language; a mix of "
        "Chinese and English is fine. Prefer vivid descriptive sentences over "
        "bare tag lists."
    ),
}

DIRECTOR_SYSTEM_PROMPT = (
    "You are an elite prompt director for AI image generation. "
    "Given a few minimal keywords, you expand them into a complete, "
    "production-ready shot: subject, scene, mood, composition, camera and "
    "lighting, and you pick a sensible aspect ratio for the composition. "
    "Always respond with ONLY one valid JSON object, no other text, in the form: "
    '{"positive_prompt": "<full positive prompt in the requested style, '
    "including subject, scene, mood and camera/lighting terms like "
    "cinematic rim light / 35mm lens / f/1.4, so it works standalone>\", "
    '"negative_prompt": "<comma-separated negative tags>", '
    '"aspect_ratio": "<one value from the allowed aspect_ratio list>"}. '
    "Rules: aspect_ratio MUST be copied exactly from the allowed list "
    "(use the same W:H notation); positive_prompt and negative_prompt must "
    "be non-empty."
)


def clamp_aspect_ratio(ratio, allowed=None):
    """归一化并钳制到允许列表；非法回退 DEFAULT_RATIO。纯函数便于单测。"""
    if allowed is None:
        allowed = ASPECT_RATIOS
    if not isinstance(ratio, str):
        return DEFAULT_RATIO
    cleaned = ratio.strip().replace("：", ":").replace("/", ":")
    cleaned = re.sub(r"\s+", "", cleaned)
    if not cleaned:
        return DEFAULT_RATIO
    for item in allowed:
        if cleaned == item.replace(" ", ""):
            return item
    # 宽高数值相等但写法不同（如 "16.0:16.0"）
    match = re.match(r"^(\d+(?:\.\d+)?):(\d+(?:\.\d+)?)$", cleaned)
    if match:
        w, h = float(match.group(1)), float(match.group(2))
        if w > 0 and h > 0:
            for item in allowed:
                iw, ih = item.split(":")
                if abs(w / h - float(iw) / float(ih)) < 1e-6:
                    return item
    return DEFAULT_RATIO


def _clean_text(text):
    """去掉多余逗号/空格，与 Style Selector 清理风格一致。"""
    if not isinstance(text, str):
        return ""
    text = text.strip().strip(",").strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"(,\s*){2,}", ", ", text)
    return text.strip().strip(",").strip()


def parse_director_json(raw, allowed=None):
    """解析导演 JSON，返回 (positive, negative, aspect_ratio)。

    解析失败时 positive 带错误前缀，其余字段给安全默认值。纯函数便于单测。
    """
    if allowed is None:
        allowed = ASPECT_RATIOS
    parsed = _extract_json_object(raw)
    if parsed is None:
        snippet = (raw or "").strip()[:200]
        return (f"API Error: director did not return valid JSON: {snippet}",
                "", DEFAULT_RATIO)
    positive = _clean_text(str(parsed.get("positive_prompt", "") or ""))
    negative = _clean_text(str(parsed.get("negative_prompt", "") or ""))
    ratio = clamp_aspect_ratio(parsed.get("aspect_ratio", ""), allowed)
    if not positive:
        snippet = (raw or "").strip()[:200]
        return (f"API Error: director returned empty positive_prompt: {snippet}",
                negative, ratio)
    return (positive, negative, ratio)


def build_user_message(keywords, model_style, extra_notes=""):
    """构造 user 消息：关键词 + 模型风格规则 + 比例白名单。纯函数便于单测。"""
    style_rule = STYLE_RULES.get(model_style, STYLE_RULES["Flux"])
    ratio_list = ", ".join(ASPECT_RATIOS)
    message = (
        f"Target model style: {model_style}\n"
        f"{style_rule}\n"
        f"Allowed aspect_ratio values: {ratio_list}\n"
        f"Minimal keywords: {keywords}"
    )
    notes = (extra_notes or "").strip()
    if notes:
        message += f"\nDirector notes (must follow): {notes}"
    return message


class FeiFeiPromptDirector:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "keywords": (
                    "STRING",
                    {"multiline": True, "default": "赛博朋克, 雨夜, 红发少女"},
                ),
                "model_style": (MODEL_STYLES, {"default": "Flux"}),
                "api_base": ("STRING", {"default": "http://127.0.0.1:8080"}),
                "api_key": ("STRING", {"default": "", "multiline": False}),
                "model": ("STRING", {"multiline": False, "default": ""}),
                "temperature": (
                    "FLOAT",
                    {"default": 0.7, "min": 0.1, "max": 1.5, "step": 0.05},
                ),
                "thinking_mode": (THINKING_MODES, {"default": THINKING_OURS}),
            },
            "optional": {
                "extra_notes": (
                    "STRING",
                    {"multiline": True, "default": ""},
                ),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("positive_prompt", "negative_prompt", "aspect_ratio")
    FUNCTION = "direct"
    CATEGORY = "FeiFei"

    def direct(self, keywords, model_style, api_base, api_key, model,
               temperature, thinking_mode, extra_notes=""):
        kw = (keywords or "").strip()
        if not kw:
            return ("API Error: keywords 为空，请输入至少一个极简关键词",
                    "", DEFAULT_RATIO)

        base = (api_base or "").strip().rstrip("/")
        if not base:
            return ("API Error: api_base is empty", "", DEFAULT_RATIO)

        user_message = build_user_message(kw, model_style, extra_notes)
        enable_thinking = thinking_mode in (THINKING_MODEL, THINKING_BOTH)
        payload = {
            "messages": [
                {"role": "system", "content": DIRECTOR_SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            "temperature": float(temperature),
            "stream": False,
            "enable_thinking": enable_thinking,
        }
        model_name = (model or "").strip()
        if model_name:
            payload["model"] = model_name

        raw_content = ""
        try:
            res_json = _post_chat_completions(base, payload, timeout=120, api_key=api_key)
            choices = res_json.get("choices") if isinstance(res_json, dict) else None
            if choices and isinstance(choices[0], dict):
                message = choices[0].get("message", {}) or {}
                raw_content = _coerce_text(message.get("content", ""))
            if not raw_content:
                raise ValueError("LLM response missing choices[0].message.content")
        except Exception as e:
            return (f"API Error: {e}", "", DEFAULT_RATIO)

        positive, negative, ratio = parse_director_json(raw_content)
        return (positive, negative, ratio)


NODE_CLASS_MAPPINGS = {"FeiFeiPromptDirector": FeiFeiPromptDirector}
NODE_DISPLAY_NAME_MAPPINGS = {"FeiFeiPromptDirector": "Prompt Director"}
