# AGENTS.md — ComfyUI-FeiFei 协作约定

## 语言

- **项目一律用英文**：代码、标识符、注释、docstring、节点显示名、widget 名称与 tooltip、面向用户的报错信息、README、commit 信息，全部英文。
- 中文只出现在跟用户的对话里，不要写进仓库文件。
- 注释只写代码本身说不清的东西，别逐行翻译代码。

## 节点写法（必须遵守）

- 统一用**经典风格**：`@classmethod INPUT_TYPES` + `RETURN_TYPES / RETURN_NAMES` + `FUNCTION` + `CATEGORY = "FeiFei"`。不要混用 `comfy_api` 的 `io.ComfyNode` 新 API。
- 每个节点文件末尾自带 `NODE_CLASS_MAPPINGS / NODE_DISPLAY_NAME_MAPPINGS`；顶层 `__init__.py` 用已有的 `try/except + _register` 容错链逐个导入，**单个节点失败不许拖死整包**。
- `folder_paths` 必须**懒加载**（函数内 `import` + `try/except` 降级），保证包在 ComfyUI 之外也能 `import` / 单测。
- 对外网络调用**优先 openai SDK**（`llm_common` 内 lazy import，节点加 `api_key` 输入、留空读 `OPENAI_API_KEY`）；未安装或客户端构造失败时自动兜底标准库 `urllib`（补 `Authorization` 头），保证包在任何环境都能 `import`。除 `openai` 外不新增第三方依赖。
- 面板顺序按使用顺序排：总开关在前、跟它绑定的一组连着排、细调项留 `optional` 并按流水线分段；函数签名顺序与 `INPUT_TYPES` 保持一致。

## 鲁棒性要求

- 所有外部输入（LLM 回包、PROMPT dict、图片 tensor 形状）都要做类型/缺失校验，失败返回错误字符串，**不许抛异常炸工作流**（除 `IMAGE` 类型错误这类 fail-fast）。
- 新增文件读写必须防目录穿越（参考 `_sanitize_subdir`），批量写文件必须保证文件名唯一。
- 尺寸相关计算统一对齐到 16 的倍数。

## 验证流程

- 改完先跑：`ComfyUI/.venv/bin/python -m py_compile <改动文件>`。
- 逻辑函数写成**纯函数**（如 `calc_size / _extract_summary / parse_wh_ratio / _normalize_base / clamp_aspect_ratio`）以便脱离 ComfyUI 单测；网络相关用本地 mock server 测，不许依赖 `:8080` 真服务。
- 用 venv 实测包导入：`sys.path.insert(0,'custom_nodes')` 后 `import_module('ComfyUI-FeiFei')`，确认 `NODE_CLASS_MAPPINGS` 包含全部节点（当前 9 个）。
- 不要提交 `__pycache__` / `*.pyc`（已有部分 `.pyc` 被误跟踪，顺手 `git checkout` 还原，别扩大）。
- commit 信息用英文短句，照现有历史的前缀风格（`Fix ...` / `Add ...` / `Support ...` / `Remove ...` / `Update ...`）；只有用户明确说 push 才提交推送。
