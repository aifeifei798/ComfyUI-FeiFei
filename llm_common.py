"""LLM 通用小模块：思维链模式常量、文本归一化、chat 请求。

qwen_prompt_node / image_caption_node / prompt_director_node 都从这里 import，
避免节点间直连——单个节点文件损坏只影响自己。

chat 请求优先走 openai SDK（可接云端 API），未安装或客户端构造失败时
自动兜底回标准库 urllib（补 Authorization 头），包在任何环境都能 import。
"""

import json
import os
import re
import urllib.error
import urllib.request
from contextlib import contextmanager

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


def _normalize_base(base):
    """归一化 api_base：去尾斜杠、确保以 /v1 结尾。纯函数便于单测。

    主机根地址（http://127.0.0.1:8080）和完整 /v1 结尾都能填，
    统一成 SDK base_url 需要的形态。
    """
    b = (base or "").strip().rstrip("/")
    if not b:
        return b
    if b.endswith("/v1"):
        return b
    return b + "/v1"


def _strip_v1(base):
    """去掉尾部 /v1，还原主机根地址（llama.cpp 原生 /completion 用）。"""
    b = (base or "").strip().rstrip("/")
    if b.endswith("/v1"):
        return b[: -len("/v1")].rstrip("/")
    return b


@contextmanager
def _safe_proxy_env():
    """临时剔除 httpx 不认的代理 scheme（如 ALL_PROXY=socks://...）。

    构造 openai 客户端时 httpx 会读快照，构造完即恢复 os.environ，
    不影响后续 urllib 等对环境的读取。合法的 http:// 代理保留。
    """
    bad_keys = []
    for key in ("ALL_PROXY", "all_proxy", "HTTP_PROXY", "http_proxy",
                "HTTPS_PROXY", "https_proxy"):
        val = os.environ.get(key)
        if val and not val.lower().startswith(("http://", "https://")):
            bad_keys.append(key)
    if not bad_keys:
        yield
        return
    saved = {k: os.environ.pop(k) for k in bad_keys}
    try:
        yield
    finally:
        os.environ.update(saved)


def _import_openai():
    """lazy import openai SDK；未安装抛 ImportError（便于测试 monkeypatch）。"""
    from openai import BadRequestError, OpenAI
    return OpenAI, BadRequestError


def _resolve_api_key(api_key):
    """节点输入优先，其次环境变量 OPENAI_API_KEY，SDK 本地场景用 EMPTY 占位。"""
    key = (api_key or "").strip()
    if key:
        return key
    return os.environ.get("OPENAI_API_KEY", "").strip()


def _make_openai_client(base, api_key, timeout):
    OpenAI, _ = _import_openai()
    key = _resolve_api_key(api_key) or "EMPTY"
    with _safe_proxy_env():
        return OpenAI(
            base_url=_normalize_base(base),
            api_key=key,
            timeout=timeout,
            max_retries=2,
        )


def _urllib_chat(base, payload, timeout=120, api_key=None):
    """标准库兜底：POST /v1/chat/completions 并解析 JSON（带 Authorization 头）。"""
    url = _normalize_base(base) + "/chat/completions"
    headers = {"Content-Type": "application/json"}
    key = _resolve_api_key(api_key)
    if key:
        headers["Authorization"] = f"Bearer {key}"

    def _send(p):
        req = urllib.request.Request(
            url, data=json.dumps(p).encode("utf-8"), headers=headers,
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


def _sdk_chat(client, payload, BadRequestError):
    """走 openai SDK 发 chat 请求，回包归一化成 urllib 同款 dict 形状。

    enable_thinking 不是 SDK 认识的字段，挪进 extra_body；
    严格服务端报 400 时去掉它重试一次。
    """
    params = dict(payload)
    # SDK v3 强制要求 model 参数；本地 llama.cpp 场景节点默认留空，补空串占位
    if not params.get("model"):
        params["model"] = ""
    enable_thinking = params.pop("enable_thinking", None)
    if enable_thinking is not None:
        params["extra_body"] = {"enable_thinking": enable_thinking}

    def _create(p):
        return client.chat.completions.create(**p)

    try:
        resp = _create(params)
    except BadRequestError as e:
        msg = str(e).lower()
        retryable = (
            enable_thinking is not None
            and any(k in msg for k in (
                "enable_thinking", "unknown", "unrecognized", "unexpected", "additional",
            ))
        )
        if not retryable:
            raise
        print("[LLM] server rejected enable_thinking, retrying without it.")
        params.pop("extra_body", None)
        resp = _create(params)

    if hasattr(resp, "model_dump"):
        data = resp.model_dump(mode="json")
    else:  # 极老版本回退
        data = json.loads(resp.json())
    # SDK 类型没显式建模的字段（如 reasoning_content）若被丢掉，从原始对象补回
    try:
        msg_obj = resp.choices[0].message
        raw_dict = data["choices"][0]["message"]
        if "reasoning_content" not in raw_dict:
            rc = getattr(msg_obj, "reasoning_content", None)
            if rc is None:
                extra = getattr(msg_obj, "model_extra", None) or {}
                rc = extra.get("reasoning_content")
            if rc is not None:
                raw_dict["reasoning_content"] = rc
    except Exception:
        pass
    return data


def _post_chat_completions(base, payload, timeout=120, api_key=None):
    """POST /v1/chat/completions 并解析 JSON。

    优先 openai SDK；未装 SDK 或客户端构造失败（如代理环境异常）时
    兜底走 urllib。请求阶段的异常一律抛给调用方（节点转错误字符串）。
    """
    client = None
    BadRequestError = None
    try:
        OpenAI, BadRequestError = _import_openai()
        client = _make_openai_client(base, api_key, timeout)
    except ImportError:
        pass  # 未安装 openai → urllib 兜底
    except Exception as e:
        # 构造失败（代理环境等）→ urllib 兜底；请求错误不在这里抛
        print(f"[LLM] openai client init failed ({e}), falling back to urllib.")

    if client is not None:
        return _sdk_chat(client, payload, BadRequestError)
    return _urllib_chat(base, payload, timeout=timeout, api_key=api_key)
