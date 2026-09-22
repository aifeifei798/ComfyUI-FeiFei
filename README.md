# ComfyUI-FeiFei

妃妃的 ComfyUI 自定义节点包（`CATEGORY = FeiFei`）：提示词增强、图生词、宽高比、水印、格式转换、WebP 存取。

## 安装

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/aifeifei798/ComfyUI-FeiFei.git
# 重启 ComfyUI，节点出现在 FeiFei 分类下
```

依赖：ComfyUI 自带 `torch / numpy / Pillow` 即可，无第三方依赖。

## 节点一览

| 节点 | 文件 | 说明 |
|---|---|---|
| Qwen-Image Prompt Enhancer (LLaMA) | `qwen_prompt_node.py` | 提示词扩写：调 OpenAI 兼容接口（llama.cpp `:8080`），输出改写后提示词 + 宽高比 + 宽高。支持 `/v1/chat/completions` 与 `/completion` 双接口，T2I / I2I 两种 system prompt |
| 图生制作词 (Image Captioner) | `image_caption_node.py` | 上传图片 → 视觉模型生成中文描述 + 英文绘画提示词。**要求接口背后是视觉模型**（如 Qwen-VL / MiniCPM-V），纯文本模型会报错 |
| 宽高比尺寸 (Aspect 1024) | `aspect_ratio_node.py` | 替代官方 Resolution Selector 的心算：选比例 + 锁定模式（固定短边/宽/高）+ 基准边（默认 1024），输出宽高（16 倍数），直连 EmptyLatent |
| 图像水印 (Watermark) | `watermark_node.py` | 三行右下角水印，字号独立可调，白色描边字，跨平台字体自动查找（`FEIFEI_FONT_PATH` 环境变量优先） |
| Image To RGB (Force 3-Channel) | `image_to_rgb.py` | 强制转 3 通道 RGB + `contiguous()`，兼容 NCHW 输入与灰度/RGBA，服务 NVIDIA RTX VSR 等挑剔的下游 |
| 风格选择器扩展版 | `style_selector_node_zh_ex.py` | prompt1/2/3 拼接 → 套角色模板 → 套风格模板，支持随机风格 |
| Save WebP (Timestamp) | `ComfyUI_SaveWebP/save_webp_node.py` | 按时间戳存 WebP（毫秒+序号防覆盖），**提示词与种子自动写入 EXIF + 同名 `.json`**（见下），`lossless` 分支、`embed_metadata` / `save_json` 开关 |
| 读取 WebP 信息 (Load WebP Info) | 同上 | 读回存入的提示词/种子：sidecar JSON 优先，EXIF 兜底；输出 positive / negative / seeds / info_json |

## WebP 元数据说明

- 存：`Save WebP` 经 hidden `PROMPT` 自动拿全工作流提示词，从 `CLIPTextEncode.text` 提正/负提示词，从 `seed` / `noise_seed` 提种子；摘要进图内 EXIF `ImageDescription`，完整 prompt + workflow 进同名 `.json`。
- 读：`PIL.Image.open(p).getexif()[270]` 或 `exiftool -ImageDescription xxx.webp`；最省事直接用`读取 WebP 信息`节点。
- 注意：ComfyUI 原生拖图还原 workflow 只认 PNG，WebP 的 EXIF 是存档可查；老图/外部图无元数据，读取节点返回空串+说明。

## LLM 接口要求

- 提示词增强：`:8080` 跑任意 OpenAI 兼容 LLM 即可。
- 图生制作词：`:8080` 必须跑**视觉模型**，`model` 输入留空（llama.cpp）或填模型名（vLLM/Ollama 类服务）。

## License

见 `LICENSE`。
