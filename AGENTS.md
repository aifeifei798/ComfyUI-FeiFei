# AGENTS.md — ComfyUI-FeiFei working conventions

## Language

- **Everything in the project is English**: code, identifiers, comments, docstrings, node display names, widget names and tooltips, user-facing error messages, README, and commit messages.
- Chinese belongs only in conversation with the user, never in repo files.
- Comments should explain what the code itself cannot; do not translate code line by line.

## Node conventions (mandatory)

- Use the **classic style** throughout: `@classmethod INPUT_TYPES` + `RETURN_TYPES / RETURN_NAMES` + `FUNCTION` + `CATEGORY = "FeiFei"`. Do not mix in the newer `comfy_api` / `io.ComfyNode` API.
- Each node file ends with its own `NODE_CLASS_MAPPINGS / NODE_DISPLAY_NAME_MAPPINGS`; the top-level `__init__.py` imports them one by one through the existing `try/except + _register` chain, so **one failing node must never take the whole pack down**.
- `folder_paths` must be **lazily loaded** (function-local `import` plus a `try/except` fallback) so the package stays importable and unit-testable outside ComfyUI.
- Prefer the **openai SDK** for outbound calls (lazy import inside `llm_common`); fall back to standard-library `urllib` (adding the `Authorization` header) when it is not installed or the client fails to build, so the package imports in any environment. Add no third-party dependency beyond `openai`.
- Endpoint settings live only in `config.json`, read through `llm_common.resolve_endpoint()`. The file is re-read on every run, so edits take effect with no ComfyUI restart. Commit the file with empty keys so a fresh checkout stays credential-free.
- **Read no environment variable at all, and never add an `api_key` / `api_base` / `model` widget.** `config.json` is the only source of settings; `os.environ` / `os.getenv` must not appear in this pack. An ambient `OPENAI_API_KEY` would be handed to whatever host a workflow names, and an ambient `OPENAI_BASE_URL` could silently redirect every request. Widgets are equally banned: ComfyUI serializes widget values into every saved workflow and image metadata as a *positional array with no field names*, so a key typed into a panel lands on disk, travels with any shared workflow or exported image, and cannot be scrubbed afterwards.
- Do not paper over an httpx/SDK build failure by mutating `os.environ` to make it pass. Let `_post_chat_completions` fall back to urllib; that is what the fallback is for.
- Order the panel by the order things get used: master switch first, the group bound to it next, fine-tuning left `optional` and segmented along the pipeline; keep the function signature order aligned with `INPUT_TYPES`.

## Robustness requirements

- Validate type and presence on every external input (LLM replies, PROMPT dicts, image tensor shapes); on failure return an error string and **never raise into the workflow** (except fail-fast cases such as a wrong `IMAGE` type).
- New file reads and writes must prevent directory traversal (see `_sanitize_subdir`); batch writes must guarantee unique file names.
- Align every size-related calculation to a multiple of 16.

## Verification workflow

- After a change, first run: `ComfyUI/.venv/bin/python -m py_compile <changed files>`.
- Write logic helpers as **pure functions** (e.g. `calc_size / _extract_summary / parse_wh_ratio / _normalize_base / clamp_aspect_ratio`) so they can be tested without ComfyUI; test network paths against a local mock server, never against a real service on `:8080`.
- Verify the package import against the venv: `sys.path.insert(0,'custom_nodes')` then `import_module('ComfyUI-FeiFei')`, and confirm `NODE_CLASS_MAPPINGS` holds every node (15 today).
- Do not commit `__pycache__` / `*.pyc` (a few `.pyc` files are already tracked by accident; `git checkout` them back rather than letting the mess spread).
- Commit messages are short English sentences following the existing prefix style (`Fix ...` / `Add ...` / `Support ...` / `Remove ...` / `Update ...`); only commit and push when the user explicitly asks.
