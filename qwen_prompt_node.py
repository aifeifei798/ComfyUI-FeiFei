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
                "mode": (["T2I (文生图)", "I2I (图生图)"], {"default": "T2I (文生图)"}),
                "api_base": ("STRING", {"default": "http://127.0.0.1:8080"}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.1, "max": 1.5, "step": 0.05}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "INT", "INT", "STRING")
    RETURN_NAMES = ("rewritten_prompt", "wh_ratio", "width", "height", "ratio_follow")
    FUNCTION = "enhance_prompt"
    CATEGORY = "FeiFei"

    def load_system_prompt(self, mode):
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
            print(f"[QwenImagePromptEnhancer] system prompt 读取失败 {filepath}: {e}")
        return "You are an expert at enhancing image prompts. Output valid JSON."

    def enhance_prompt(self, user_prompt, mode, api_base, temperature):
        system_prompt = self.load_system_prompt(mode)
        base = (api_base or "").strip().rstrip("/")
        if not base:
            return ("API Error: api_base 为空", "", DEFAULT_W, DEFAULT_H, "")
        url = base + "/v1/chat/completions"

        payload = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": temperature,
            "stream": False
        }

        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})

        raw_content = ""
        first_error = ""
        try:
            with urllib.request.urlopen(req, timeout=120) as response:
                body = response.read().decode("utf-8", errors="replace")
                res_json = json.loads(body)
                # OpenAI 兼容格式：choices[0].message.content
                choices = res_json.get("choices") if isinstance(res_json, dict) else None
                if choices:
                    message = choices[0].get("message", {}) if isinstance(choices[0], dict) else {}
                    raw_content = (message.get("content") or "").strip()
                if not raw_content:
                    raise ValueError(f"LLM 返回缺少 choices[0].message.content: {body[:500]}")
        except Exception as e:
            first_error = str(e)
            # 兼容 llama.cpp 原生 /completion 接口
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
                    if not raw_content:
                        raise ValueError(f"/completion 返回缺少 content: {str(resp_body)[:500]}")
            except Exception as ex:
                return (f"API Error: {first_error} / {str(ex)}", "", DEFAULT_W, DEFAULT_H, "")

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

        return (rewritten_prompt, wh_ratio, width, height, ratio_follow)