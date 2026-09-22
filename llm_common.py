"""LLM 通用小模块：思维链模式常量、文本归一化、chat 请求。

qwen_prompt_node 与 image_caption_node 都从这里 import，
避免 caption 直连 qwen——单个节点文件损坏只影响自己。
"""

import json
import re
import urllib.error
import urllib.request

# 思维链模式：Ours = 本包定义的链（8 步 prompt / 自定义指令）；
# Model native = 极简 prompt 走模型自带思考；Both = 都要
THINKING_OURS = "Ours (8-step)"
THINKING_MODEL = "Model native"
THINKING_BOTH = "Both"
THINKING_MODES = [THINKING_OURS, THINKING_MODEL, THINKING_BOTH]


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
