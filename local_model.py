"""Shared local HF backend: load one HF-format folder with transformers and run
chat generation on it, for every node that offers a Local HF source.

Nothing here downloads. A folder without config.json and a weight file reports
the `hf download` command instead, so the model arrives the way the user chose
to fetch it and ComfyUI never starts a download on its own.

transformers is imported inside the loading function, so the pack still imports
(and the API path still runs) where transformers is missing or too old.
"""

import gc
import os
import threading

# LFM3-style configs trip over rope standardization before this version
REQUIRED_TRANSFORMERS = "5.21"

WEIGHT_SUFFIXES = (".safetensors", ".bin", ".pt", ".pth")

_LOCK = threading.Lock()
_MODELS = {}

# Nodes without a max_tokens widget leave this cap in place
LOCAL_MAX_TOKENS = 2048


def resolve_model_dir(model_dir):
    """Widget value -> absolute path; an empty or blank widget stays empty."""
    text = (model_dir or "").strip()
    if not text:
        return ""
    return os.path.abspath(os.path.expanduser(text))


def model_files_ready(model_dir):
    """True when the folder exists and holds a config.json plus a weight file."""
    if not model_dir or not os.path.isdir(model_dir):
        return False
    try:
        entries = os.listdir(model_dir)
    except OSError:
        return False
    names = [name.lower() for name in entries]
    return "config.json" in names and any(n.endswith(WEIGHT_SUFFIXES) for n in names)


def build_download_command(hf_model, model_dir):
    """The command the user runs themselves; the node never downloads anything."""
    repo = (hf_model or "").strip() or "<hf-repo-id>"
    target = (model_dir or "").strip()
    if not target:
        # No folder configured means the Hub cache is the place to put it
        return f"hf download {repo}"
    return f'hf download {repo} --local-dir "{target}"'


def hf_cache_path(hf_model):
    """Cached Hub snapshot path for a repo id, or "" when it is not cached.

    local_files_only keeps this offline: an uncached repo returns "" instead of
    starting a download.
    """
    repo = (hf_model or "").strip()
    if not repo:
        return ""
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        return ""
    try:
        return snapshot_download(repo_id=repo, local_files_only=True) or ""
    except Exception:
        return ""


def resolve_model_source(hf_model, model_dir):
    """Where the Local HF source loads from: (kind, path), or ("", "") when neither works.

    model_dir wins whenever it holds a model. hf_model is the fallback and only
    counts once a snapshot of that repo sits in the Hub cache, so exactly one of
    the two keys is used per run and neither ever downloads anything.
    """
    folder = resolve_model_dir(model_dir)
    if model_files_ready(folder):
        return "dir", folder
    cached = hf_cache_path(hf_model)
    if cached and model_files_ready(cached):
        return "hf_cache", cached
    return "", ""


def missing_model_hint(hf_model, model_dir):
    folder = resolve_model_dir(model_dir)
    if folder:
        where = (f"Local model not found: {folder}\n"
                 f"Expected config.json plus a weight file ({', '.join(WEIGHT_SUFFIXES)}) in that folder.")
    else:
        where = ("Local model not found: model_dir is empty and hf_model has no cached snapshot.")
    return (
        f"{where}\n"
        "Download it, then run this node again:\n"
        f"  {build_download_command(hf_model, folder)}\n"
        f"Loading it needs transformers>={REQUIRED_TRANSFORMERS} in the ComfyUI environment."
    )


def _release_other_models(keep):
    """Drop every cached model except `keep`, then hand the VRAM back."""
    dropped = [key for key in _MODELS if key != keep]
    for key in dropped:
        _MODELS.pop(key, None)
    if dropped:
        print(f"[FeiFei local] releasing local model: {', '.join(dropped)}")
        gc.collect()
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass


def _load_model(model_dir):
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    if torch.cuda.is_available():
        device = "cuda"
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    else:
        device = "cpu"
        dtype = torch.float32
    print(f"[FeiFei local] loading {model_dir} on {device} ({dtype}) ...")
    model = AutoModelForImageTextToText.from_pretrained(
        model_dir,
        dtype=dtype,
        trust_remote_code=True,
    ).eval().to(device)
    processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True)
    return model, processor


def get_model(model_dir):
    """Cached (model, processor) for this folder; one folder is kept at a time."""
    with _LOCK:
        _release_other_models(keep=model_dir)
        loaded = _MODELS.get(model_dir)
        if loaded is None:
            loaded = _load_model(model_dir)
            _MODELS[model_dir] = loaded
        return loaded


def run_chat(node_class, system_prompt, user_content, temperature, enable_thinking=False,
             max_tokens=LOCAL_MAX_TOKENS):
    """One local chat turn for a node class, using the model from config.json.

    Returns (content, thinking, error): error is the download hint when no model
    is configured, "Local Error: ..." when loading or generating fails, and ""
    on success. Reasoning left in the reply is split off or stripped, so content
    only ever holds the answer.
    """
    from .llm_common import _split_think_tags, _strip_thinking, resolve_local_model

    hf_model, model_dir = resolve_local_model(node_class)
    kind, folder = resolve_model_source(hf_model, model_dir)
    if not folder:
        hint = missing_model_hint(hf_model, model_dir)
        print(f"[{node_class}] {hint}")
        return "", "", hint
    print(f"[{node_class}] local source: {kind} -> {folder}")

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]
    try:
        text = generate(folder, messages, temperature=temperature, max_tokens=max_tokens,
                        enable_thinking=enable_thinking)
    except ImportError as e:
        return "", "", (f'Local Error: {e}; install it with: '
                        f'pip install -U "transformers>={REQUIRED_TRANSFORMERS}"')
    except Exception as e:
        return "", "", f"Local Error: {e}"

    thinking, content = _split_think_tags(text)
    content = _strip_thinking(content)
    if not content:
        return "", "", "Local Error: model returned an empty reply"
    return content, thinking, ""


def generate(model_dir, messages, temperature=0.3, max_tokens=1024, enable_thinking=False):
    """Run one chat completion on the local folder and return the reply text.

    `messages` use the multimodal content form: a plain string, or a list of
    {"type": "text"} / {"type": "image", "image": PIL} parts. Chat templates that
    do not know enable_thinking simply ignore it. Raises on any failure.
    """
    import torch

    model, processor = get_model(model_dir)
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        enable_thinking=enable_thinking,
    )
    inputs = {k: (v.to(model.device) if torch.is_tensor(v) else v) for k, v in inputs.items()}

    kwargs = {"max_new_tokens": max(16, int(max_tokens)), "repetition_penalty": 1.1}
    if temperature and temperature > 0:
        kwargs.update(do_sample=True, temperature=float(temperature), top_k=50)
    else:
        kwargs["do_sample"] = False

    output = model.generate(**inputs, **kwargs)
    prompt_len = inputs["input_ids"].shape[-1]
    return processor.tokenizer.decode(output[0][prompt_len:], skip_special_tokens=True)
