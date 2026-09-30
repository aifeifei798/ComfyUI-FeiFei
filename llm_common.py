"""Shared LLM helpers: thinking-mode constants, text normalization, chat requests.

qwen_prompt_node / image_caption_node / prompt_director_node all import from
here so the nodes do not depend on each other - one broken node file then only
breaks itself.

Chat requests prefer the openai SDK (which can reach cloud APIs) and fall back to
standard-library urllib (adding the Authorization header) when the SDK is missing
or the client fails to build, so the pack imports in any environment.
"""

import json
import os
import re
import urllib.error
import urllib.request
from contextlib import contextmanager

# Thinking modes: Ours = the chain defined by this pack (8-step prompt / custom
# instructions); Model native = minimal prompt that leans on the model's own
# thinking; Both = run each
THINKING_OURS = "Ours (8-step)"
THINKING_MODEL = "Model native"
THINKING_BOTH = "Both"
THINKING_MODES = [THINKING_OURS, THINKING_MODEL, THINKING_BOTH]


def _coerce_text(value):
    """content/reasoning may be a string or a list of parts; normalize to a string"""
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
    """Strip a ```json ... ``` / ``` ... ``` wrapper and return the inner text"""
    if not isinstance(text, str):
        return ""
    stripped = text.strip()
    # ```json\n{...}\n``` or ```\n{...}\n```
    fence_match = re.search(r"```(?:json)?\s*(.*?)```", stripped, re.DOTALL | re.IGNORECASE)
    if fence_match:
        return fence_match.group(1).strip()
    return stripped


def _extract_json_object(text):
    """Robustly pull the first parseable JSON object out of LLM text.

    Tries a whole-string parse first, then scans brace-balanced candidates so a
    greedy regex cannot span several objects. Returns a dict or None.
    """
    cleaned = _strip_code_fences(text)
    if not cleaned:
        return None
    # 1) The whole string is JSON
    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass
    # 2) Brace-balanced scan: from each '{' forward to its matching '}'
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
    # Try the longest candidate first (usually the one we want)
    for cand in sorted(candidates, key=len, reverse=True):
        try:
            parsed = json.loads(cand)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            continue
    return None


def _normalize_base(base):
    """Normalize api_base: drop the trailing slash and ensure it ends with /v1.

    Pure function for unit tests. Both a host root (http://127.0.0.1:8080) and a
    full /v1 URL are accepted and collapse into the shape the SDK base_url wants.
    """
    b = (base or "").strip().rstrip("/")
    if not b:
        return b
    if b.endswith("/v1"):
        return b
    return b + "/v1"


def _strip_v1(base):
    """Drop the trailing /v1 to recover the host root (used by llama.cpp /completion)."""
    b = (base or "").strip().rstrip("/")
    if b.endswith("/v1"):
        return b[: -len("/v1")].rstrip("/")
    return b


@contextmanager
def _safe_proxy_env():
    """Temporarily drop proxy schemes httpx rejects (e.g. ALL_PROXY=socks://...).

    httpx reads a snapshot of the environment when the openai client is built, so
    os.environ is restored right after and later readers such as urllib are
    unaffected. Valid http:// proxies are kept.
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
    """Lazy import of the openai SDK; raises ImportError when missing (easy to monkeypatch in tests)."""
    from openai import BadRequestError, OpenAI
    return OpenAI, BadRequestError


def _resolve_api_key(api_key):
    """Use only the key typed into the node's api_key input; empty sends no Authorization header.

    Deliberately does not read the OPENAI_API_KEY environment variable: api_base is a
    workflow-editable widget, so an env key would be handed to whatever host the workflow
    names. Paste the key into the input if you want to use one.
    """
    return (api_key or "").strip()


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
    """Standard-library fallback: POST /v1/chat/completions and parse the JSON (with Authorization header)."""
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
    """Send the chat request through the openai SDK, normalizing the reply into the
    same dict shape the urllib path returns.

    enable_thinking is not a field the SDK models, so it moves into extra_body; if a
    strict server answers 400 it is dropped and the request is retried once.
    """
    params = dict(payload)
    # SDK v3 requires model; local llama.cpp leaves the node input empty, so fill a placeholder
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
    else:  # very old SDK fallback
        data = json.loads(resp.json())
    # Fields the SDK types do not model (reasoning_content) may be dropped, so read them off the raw object
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
    """POST /v1/chat/completions and parse the JSON.

    Prefers the openai SDK; falls back to urllib when the SDK is missing or the
    client fails to build (a broken proxy environment, say). Request-stage
    exceptions propagate to the caller, which turns them into error strings.
    """
    client = None
    BadRequestError = None
    try:
        OpenAI, BadRequestError = _import_openai()
        client = _make_openai_client(base, api_key, timeout)
    except ImportError:
        pass  # openai not installed -> urllib fallback
    except Exception as e:
        # Client construction failed (proxy environment, etc.) -> urllib fallback;
        # request errors are not raised here
        print(f"[LLM] openai client init failed ({e}), falling back to urllib.")

    if client is not None:
        return _sdk_chat(client, payload, BadRequestError)
    return _urllib_chat(base, payload, timeout=timeout, api_key=api_key)
