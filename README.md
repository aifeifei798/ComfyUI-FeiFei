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
| Film Grain & Tone | `film_grain_node.py` | 物理胶片后处理，纯 torch 无外部模型。**接在出图之后、打水印之前**。有机颗粒 + halation + 暗角色散 + 冲印曲线 + 微对比 |
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

## Film Grain & Tone（物理胶片后处理）

给 AI 图像去掉「太干净太塑料」的观感。零外部模型，只用 torch，1024² 约 0.2s、4K 约 10ms（CUDA）。

处理顺序按真实胶片流程：**镜头**（暗角 / 侧向色散）→ **胶片基**（halation 光晕）→ **冲印**（S 曲线 / 分离色调 / 微对比）→ **乳剂**（颗粒）。颗粒放最后，避免被冲印曲线压掉。

接在 `VAEDecode` 之后、`Watermark` 之前。

### 怎么选：`film_type` / `preset` / `preset_mix`

两个下拉 + 一个混合度滑杆。`film_type` 只是给 `preset` 分组用的，**它本身不产生任何效果**。

| 想要的效果 | 怎么设 |
|---|---|
| 一键胶片味 | `preset` 选任意型号，`preset_mix = 1.0`（默认） |
| 在预设基础上微调 | `preset_mix` 调到 `0.3`~`0.7`，然后拖滑杆 |
| **完全自定义** | ① `preset = custom`，或 ② `preset_mix = 0.0` —— 两种写法效果**完全相同** |

**判断是否已经是自定义**：只要 `preset` 是 `custom` **或** `preset_mix` 是 `0.0`，滑杆就 100% 接管，`film_type` 选什么都不影响结果。

> `preset = custom` 时 `preset_mix` 会被忽略（反正没有预设可混）。反之 `preset_mix = 0.0` 时 `preset` 也会被忽略。两条路殊途同归。

`preset` 只改胶片「性格」（颗粒大小、明暗响应、光晕色调、镜头衰减），**不做整体上色** —— 所以图像本身的颜色不会被预设改掉，只会被「拍在胶片上」。

改 `film_type` 而没改 `preset` 时，节点会自动回落到新类别的第一款，避免看起来「换了类别却没效果」。

> 每个控件鼠标悬停都有 tooltip，说明这个值调到多少会出什么问题（比如 `grain_amount` 超过 0.6 就开始像电视雪花）。

| film_type | 型号 |
|---|---|
| Color Negative | Portra 160 / 400 / 800、Ektar 100 / 500、Gold 200 / 400、Fuji C200、Pro 400H |
| Slide | Ektachrome E100、Fuji Provia 100F |
| B&W Negative | Tri-X 400、HP5 Plus 400、FP4 Plus 125、TMax 100 / 400、Delta 100 / 3200、Acros 100 |
| Cinema | CineStill 400D / 800T、Vision3 250D / 500T、Cine 50D、500T Expired |
| Special | Cinestack 800T、Push +2 Stops、Cross Process |
| Other | custom、Digital Clean（几乎什么都不加，当 A/B 基线用） |

每种胶片只写自己与「普通负片基准」不同的项，其余继承基准，所以调一种胶片不会牵动其他型号。

### 主要滑杆

| 控件 | 默认 | 说明 |
|---|---|---|
| `grain_amount` | 0.25 | 颗粒强度。默认约 ±3.5/255，拉满 1.0 约 ±14/255。**再往上会变成电视雪花而不是胶片** |
| `grain_size` | 1.0 | 颗粒团尺寸，以 1024 短边为基准自动缩放，4K 上颗粒物理尺寸自动变大 |
| `halation` | 0.15 | 高光光晕强度。想要「漏光」感可以拉到 0.3~0.5 |
| `vignette` | 0.12 | 暗角强度 |
| `tone` | 0.25 | 冲印 S 曲线强度，内含约 0.015 的黑位抬升 |
| `preset_mix` | 1.0 | 见上 |

折叠区还有 `grain_shadows`（颗粒往暗部倾斜的程度）、`grain_chroma`（0 = 全单色颗粒，1 = 三通道独立）、`halation_threshold`、`halation_radius`、`vignette_size`、`micro_contrast`（去塑料感最有效的一招）、`chroma_shift`（单位是 1024 画幅下的像素偏移）、`split_tone`。

### 颗粒为什么不像椒盐噪点

真实胶片的银盐颗粒是**成团**的，不是逐像素独立的。所以噪点在低分辨率网格上生成后 bicubic 上采样，只混 6% 的逐像素细节；同时按亮度加权 —— 中调最强、亮部快速衰减、暗部按 `grain_shadows` 倾斜。逐像素细节占比是关键，调高就会立刻退化成数字噪点。

### 注意

- 颗粒低于 1/255 会被 8-bit 保存或 `VAEEncode` 量化吃掉，别把 `grain_amount` 拉到 0.02 以下
- `seed` 可复现；批量输入时每帧噪声独立（`seed + 序号`），接视频首帧序列不会闪烁
- RGBA 输入只处理 RGB，alpha 原样保留；灰度 / 双通道输入也支持

## WebP metadata

- Save: full workflow prompt comes via hidden `PROMPT`; positive/negative resolved by tracing workflow links (falls back to order); seeds collected from `seed` / `noise_seed`. Summary → EXIF `ImageDescription`, full prompt + workflow → sidecar `.json`.
- Read: `PIL.Image.open(p).getexif()[270]`, `exiftool -ImageDescription xxx.webp`, or the Load WebP Info node.
- Note: ComfyUI's native drag-to-restore only understands PNG; WebP EXIF is for archiving. Old/external images return empty strings + a note.

## LLM backend requirements

- Prompt enhancing / Prompt Director: any text LLM behind an OpenAI-compatible API — local llama.cpp on `:8080` or a cloud endpoint via `api_base` + `api_key`.
- Captioning: **vision model** behind the same API; leave `model` empty (llama.cpp) or set it (vLLM/Ollama-style servers).

## License

See `LICENSE`.
