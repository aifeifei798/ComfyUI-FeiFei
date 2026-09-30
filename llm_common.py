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


def _import_openai():
    """Lazy import of the openai SDK; raises ImportError when missing (easy to monkeypatch in tests)."""
    from openai import BadRequestError, OpenAI
    return OpenAI, BadRequestError


def _resolve_api_key(api_key):
    """Normalize a key read from config.json; empty sends no Authorization header.

    The key never comes from a widget: a widget value is serialized verbatim into
    every saved workflow and into image metadata, and workflow widget values are
    stored as a positional array with no field name, so a leaked key cannot even
    be scrubbed afterwards. config.json stays outside the workflow.
    """
    return (api_key or "").strip()


CONFIG_FILENAME = "config.json"


def load_config():
    """Read config.json next to this pack. Tolerates missing/corrupt files -> {}.

    Read on every call so editing the file takes effect without restarting ComfyUI.
    config.json is the only source of endpoint settings; there is no built-in
    fallback base URL, so a missing file simply means no endpoint.
    """
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), CONFIG_FILENAME)
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"[LLM] {CONFIG_FILENAME} not found, no endpoint configured.")
        return {}
    except (OSError, ValueError, UnicodeDecodeError) as e:
        print(f"[LLM] {CONFIG_FILENAME} unreadable ({e}), no endpoint configured.")
        return {}
    return data if isinstance(data, dict) else {}


def resolve_endpoint(node_class=""):
    """Return (api_base, api_key, model) for a node from config.json.

    A per_node section for this node class overrides the top-level value; an empty
    or missing per_node entry inherits the top-level one.
    """
    cfg = load_config()
    per_node = cfg.get("per_node")
    local = per_node.get(node_class, {}) if isinstance(per_node, dict) else {}
    if not isinstance(local, dict):
        local = {}

    def pick(key):
        value = local.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
        value = cfg.get(key)
        return value.strip() if isinstance(value, str) else ""

    return pick("api_base"), pick("api_key"), pick("model")


def _make_openai_client(base, api_key, timeout):
    OpenAI, _ = _import_openai()
    return OpenAI(
        base_url=_normalize_base(base),
        api_key=_resolve_api_key(api_key) or "EMPTY",
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
