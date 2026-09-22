import os
import json
import re
import math
import urllib.request
import urllib.error

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

# 思维链模式：Ours = 本包 8 步 system prompt；Model native = 极简 prompt 走模型自带思考
THINKING_OURS = "Ours (8-step)"
THINKING_MODEL = "Model native"
THINKING_BOTH = "Both"
THINKING_MODES = [THINKING_OURS, THINKING_MODEL, THINKING_BOTH]

# 模型原生链用的极简 system prompt：只规定 JSON 格式，不给推导脚手架
NATIVE_SYSTEM_PROMPT = (
    "You are an expert at enhancing image prompts for image generation. "
    "Think freely, then output ONLY one valid JSON object, no other text: "
    '{"rewritten_prompt": "<detailed English image description>", '
    '"wh_ratio": "<e.g. 3:2, empty string if unsure>", '
    '"ratio_follow": "<notes or empty string>"}.'
)

def _strip_code_fences(text):
    """去掉 ```json ... ``` / ``` ... ``` 包裹，返回内部文本"""
    if not isinstance(text, str):
        return ""
    stripped = text.strip()
    # ```json\n{...}\n``` 或 ```\n{...}\n```
    fence_match = re.search(r"```(?:json)?\s*(.*?)```", stripped, re.DOTALL | re.IGNORECASE)
    if fence_match:
        return fence_match.group(1).strip()
    return stripped


def _extract_json_object(text):
    """从 LLM 文本中稳健提取第一个可解析的 JSON 对象。

    先尝试整体解析，再用花括号配平扫描候选（避免贪婪正则跨多个对象）。
    返回 dict 或 None。
    """
    cleaned = _strip_code_fences(text)
    if not cleaned:
        return None
    # 1) 整体就是 JSON
    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass
    # 2) 括号配平扫描：找到每个 '{' 向后配平到闭合 '}'
    candidates = []
    depth = 0
    start = -1
    in_string = False
    escape = False
    for i, ch in enumerate(cleaned):
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        else:
            if ch == '"':
                in_string = True
            elif ch == "{":
                if depth == 0:
                    start = i
                depth += 1
            elif ch == "}":
                if depth > 0:
                    depth -= 1
                    if depth == 0 and start != -1:
                        candidates.append(cleaned[start:i + 1])
                        start = -1
    # 从最长的候选开始试（通常即为目标对象）
    for cand in sorted(candidates, key=len, reverse=True):
        try:
            parsed = json.loads(cand)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            continue
    return None


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


def _coerce_text(value):
    """content/reasoning 可能是字符串或 parts 列表，统一成字符串"""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        parts = []
        for item in value:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(parts).strip()
    return ""


def _post_chat_completions(base, payload, timeout=120):
    """POST /v1/chat/completions 并解析 JSON。

    部分严格服务端会因未知字段（如 enable_thinking）报 400，
    此时去掉该字段原样重试一次；仍失败则抛异常给调用方。
    """
    url = base + "/v1/chat/completions"

    def _send(p):
        req = urllib.request.Request(
            url, data=json.dumps(p).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8", errors="replace"))

    try:
        return _send(payload)
    except urllib.error.HTTPError as e:
        body = ""
        try:
            body = e.read().decode("utf-8", errors="replace")
        except Exception:
            pass
        retryable = (
            e.code == 400
            and "enable_thinking" in payload
            and any(k in body.lower() for k in (
                "enable_thinking", "unknown", "unrecognized", "unexpected", "additional",
            ))
        )
        if not retryable:
            raise
        print("[LLM] server rejected enable_thinking, retrying without it.")
        retry_payload = {k: v for k, v in payload.items() if k != "enable_thinking"}
        return _send(retry_payload)


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
                "n_predict": 1024
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