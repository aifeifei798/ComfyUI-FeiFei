# AGENTS.md — ComfyUI-FeiFei working conventions

## Language

- **Everything in the project is English**: code, identifiers, comments, docstrings, node display names, widget names and tooltips, user-facing error messages, README, and commit messages.
- Chinese belongs only in conversation with the user, never in repo files.
- Comments should explain what the code itself cannot; do not translate code line by line.

## Node conventions (mandatory)

- Use the **classic style** throughout: `@classmethod INPUT_TYPES` + `RETURN_TYPES / RETURN_NAMES` + `FUNCTION` + `CATEGORY = "FeiFei"`. Do not mix in the newer `comfy_api` / `io.ComfyNode` API.
- Each node file ends with its own `NODE_CLASS_MAPPINGS / NODE_DISPLAY_NAME_MAPPINGS`; the top-level `__init__.py` imports them one by one through the existing `try/except + _register` chain, so **one failing node must never take the whole pack down**.
- `folder_paths` must be **lazily loaded** (function-local `import` plus a `try/except` fallback) so the package stays importable and unit-testable outside ComfyUI.
- Prefer the **openai SDK** for outbound calls (lazy import inside `llm_common`, with an `api_key` node input); fall back to standard-library `urllib` (adding the `Authorization` header) when it is not installed or the client fails to build, so the package imports in any environment. Add no third-party dependency beyond `openai`.
- **`api_key` only ever accepts a value typed into the input; never read the environment**: `api_base` is a workflow-editable widget, so falling back to `OPENAI_API_KEY` lets a shared workflow forward the user's key to any host it names.
- Order the panel by the order things get used: master switch first, the group bound to it next, fine-tuning left `optional` and segmented along the pipeline; keep the function signature order aligned with `INPUT_TYPES`.

## Robustness requirements

- Validate type and presence on every external input (LLM replies, PROMPT dicts, image tensor shapes); on failure return an error string and **never raise into the workflow** (except fail-fast cases such as a wrong `IMAGE` type).
- New file reads and writes must prevent directory traversal (see `_sanitize_subdir`); batch writes must guarantee unique file names.
- Align every size-related calculation to a multiple of 16.

## Verification workflow

- After a change, first run: `ComfyUI/.venv/bin/python -m py_compile <changed files>`.
- Write logic helpers as **pure functions** (e.g. `calc_size / _extract_summary / parse_wh_ratio / _normalize_base / clamp_aspect_ratio`) so they can be tested without ComfyUI; test network paths against a local mock server, never against a real service on `:8080`.
- Verify the package import against the venv: `sys.path.insert(0,'custom_nodes')` then `import_module('ComfyUI-FeiFei')`, and confirm `NODE_CLASS_MAPPINGS` holds every node (10 today).
- Do not commit `__pycache__` / `*.pyc` (a few `.pyc` files are already tracked by accident; `git checkout` them back rather than letting the mess spread).
- Commit messages are short English sentences following the existing prefix style (`Fix ...` / `Add ...` / `Support ...` / `Remove ...` / `Update ...`); only commit and push when the user explicitly asks.
