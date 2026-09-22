# ComfyUI-FeiFei

FeiFei's ComfyUI custom nodes (`CATEGORY = FeiFei`): LLM-driven Prompt Director, prompt enhancing, image captioning, aspect-ratio sizing, style/character template assembly, watermark, format conversion, WebP save/load.

![ComfyUI-FeiFei](images/ComfyUI-FeiFei.png)

### Example workflow

The screenshot above comes with the full workflow JSON: [`Workflow/ComfyUI-FeiFei.json`](Workflow/ComfyUI-FeiFei.json). Download it and drag-and-drop onto the ComfyUI canvas to load (missing custom nodes will show as red boxes if you haven't installed this pack).

## Install

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/aifeifei798/ComfyUI-FeiFei.git
# Restart ComfyUI, nodes appear under the FeiFei category
```

Requirements: stock ComfyUI `torch / numpy / Pillow` plus `openai>=1.40` (see `requirements.txt`). If `openai` is missing, LLM calls automatically fall back to the standard library — the pack still imports and runs.

## Nodes

| Node | File | Notes |
|---|---|---|
| Prompt Director | `prompt_director_node.py` | Type a few minimal keywords (e.g. "cyberpunk, rainy night, red-haired girl"); an LLM director expands them into a ready-to-use shot. Outputs `positive_prompt` (subject + scene + mood + camera/lighting merged), `negative_prompt`, and `aspect_ratio` (clamped to the Aspect Ratio node's whitelist — convert that node's widget to input and connect it). `model_style` switches prompt dialect: **Flux** natural language / **SDXL** comma tags / **Qwen-Image** bilingual prose. Supports `api_key` + cloud APIs |
| Qwen-Image Prompt Enhancer (LLaMA) | `qwen_prompt_node.py` | Prompt rewriting via OpenAI-compatible API (llama.cpp `:8080` by default). Inputs: `api_base` / `api_key` / `model` / `temperature` / `thinking_mode`. Outputs rewritten prompt + ratio + size + `thinking`. `/v1/chat/completions` with `/completion` fallback, T2I / I2I system prompts |
| Image Captioner | `image_caption_node.py` | Upload image → vision model writes Chinese description + English prompt. **Requires a vision model behind the API** (e.g. Qwen-VL / MiniCPM-V); `max_side` caps upload size (default 1024). `api_key` supported |
| Aspect Ratio (1024) | `aspect_ratio_node.py` | No more megapixel math: pick ratio + lock mode (Short Side / Fixed Width / Fixed Height) + base side (default 1024), outputs 16-aligned width/height straight into Empty Latent. Its `aspect_ratio` widget accepts a connected ratio string (e.g. from Prompt Director) after Convert to input |
| Watermark | `watermark_node.py` | Three-line bottom-right watermark, per-line font size, white text with black stroke, cross-platform font lookup (`FEIFEI_FONT_PATH` first) |
| Image To RGB (Force 3-Channel) | `image_to_rgb.py` | Forces 3-channel RGB + `contiguous()`, tolerates NCHW / grayscale / RGBA, for picky downstream nodes (e.g. NVIDIA RTX VSR) |
| Style Selector EX | `style_selector_node_zh_ex.py` | prompt1/2/3 text boxes + **four extra input sockets `prompt4`~`prompt7`** (link-only, appended after the boxes) → character template (`juese_data.py`) → style template (`style_data.py`), optional random style. Add styles/characters by editing the two data files |
| Save WebP (Timestamp) | `ComfyUI_SaveWebP/save_webp_node.py` | Timestamped WebP (millisecond + index, no overwrites). **Prompt + seeds auto-saved to EXIF + sidecar `.json`** (below); `lossless`, `embed_metadata` / `save_json` toggles |
| Load WebP Info | same | Reads back prompt/seeds: sidecar JSON first, EXIF fallback. Outputs positive / negative / seeds / info_json |

Shared LLM helpers (`_post_chat_completions`, thinking-mode constants, JSON extraction, base-URL normalization) live in `llm_common.py` so the LLM nodes don't depend on each other.

## LLM access: openai SDK first, urllib fallback

- LLM nodes take an `api_key` input; leave it empty to read the `OPENAI_API_KEY` environment variable. Local llama.cpp needs neither.
- `api_base` accepts a host root (`http://127.0.0.1:8080`) or a full `/v1` URL — it is normalized automatically, so cloud endpoints (OpenAI / DeepSeek / Moonshot / any OpenAI-compatible server) work the same way.
- Requests go through the official `openai` Python SDK when installed (with retries and auth handled for you). If the SDK is missing or fails to initialize, the node falls back to standard-library `urllib` with an `Authorization` header — a broken proxy env or missing package never kills the pack.
- Cloud APIs require filling the `model` input (leave empty only for llama.cpp-style servers that ignore it).

## thinking_mode (Qwen + Captioner + Prompt Director)

| Mode | System prompt | `enable_thinking` | Measured (same prompt) |
|---|---|---|---|
| `Ours (8-step)` (default) | Full 8-step file / instruction box | false | ~1600 chars content |
| `Model native` | Built-in minimal prompt (JSON keys only) | true | ~550 chars content |
| `Both` | Full prompt / instruction box | true | For future Qwen3-class models |

If the server rejects `enable_thinking` with 400, the node retries once without it. Old workflows pick up the default mode, no changes needed.

## WebP metadata

- Save: full workflow prompt comes via hidden `PROMPT`; positive/negative resolved by tracing workflow links (falls back to order); seeds collected from `seed` / `noise_seed`. Summary → EXIF `ImageDescription`, full prompt + workflow → sidecar `.json`.
- Read: `PIL.Image.open(p).getexif()[270]`, `exiftool -ImageDescription xxx.webp`, or the Load WebP Info node.
- Note: ComfyUI's native drag-to-restore only understands PNG; WebP EXIF is for archiving. Old/external images return empty strings + a note.

## LLM backend requirements

- Prompt enhancing / Prompt Director: any text LLM behind an OpenAI-compatible API — local llama.cpp on `:8080` or a cloud endpoint via `api_base` + `api_key`.
- Captioning: **vision model** behind the same API; leave `model` empty (llama.cpp) or set it (vLLM/Ollama-style servers).

## License

See `LICENSE`.
