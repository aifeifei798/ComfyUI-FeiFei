"""Subtitle translator node: bilingual subtitles for the cinematic frame.

Takes the two subtitle lines and translates them to the target language via
the OpenAI-compatible endpoint in config.json, so a Chinese line can ride
under the English original into Cinematic Frame & Subtitle. Failures return
an error string instead of raising.
"""

from .llm_common import (
    _coerce_text,
    _extract_json_object,
    _post_chat_completions,
    resolve_endpoint,
)

TARGET_LANGS = ["English", "Chinese"]

TRANSLATE_SYSTEM = (
    "You are a professional film-subtitle translator. Translate each line "
    "into the target language, keeping the tone concise and cinematic, one "
    "line per line, no explanations. Always respond with ONLY one valid JSON "
    'object: {"top": "<translated first line>", "bottom": "<translated '
    'second line>"}. When a line is empty, keep it empty.'
)


def build_translate_messages(top, bottom, target, extra_notes=""):
    """Build the user message. Pure function for tests."""
    message = (
        f"Target language: {target}\n"
        f"Line 1: {top or ''}\n"
        f"Line 2: {bottom or ''}"
    )
    notes = (extra_notes or "").strip()
    if notes:
        message += f"\nNotes (must follow): {notes}"
    return message


def parse_translate_json(raw):
    """Parse the translator JSON into (top, bottom). Pure function for tests."""
    parsed = _extract_json_object(raw)
    if not isinstance(parsed, dict):
        snippet = (raw or "").strip()[:200]
        return (f"API Error: translator did not return valid JSON: {snippet}", "")
    top = parsed.get("top", "")
    bottom = parsed.get("bottom", "")
    top = top.strip() if isinstance(top, str) else ""
    bottom = bottom.strip() if isinstance(bottom, str) else ""
    return (top, bottom)


class FeiFeiSubtitleTranslator:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text_top": ("STRING", {"multiline": True, "default": ""}),
                "text_bottom": ("STRING", {"multiline": True, "default": ""}),
                "target_lang": (TARGET_LANGS, {"default": "Chinese"}),
                "temperature": ("FLOAT", {"default": 0.3, "min": 0.1, "max": 1.5, "step": 0.05}),
            },
            "optional": {
                "extra_notes": ("STRING", {"multiline": True, "default": ""}),
                "timeout": ("INT", {
                    "default": 120, "min": 10, "max": 600, "step": 5,
                    "tooltip": "Request timeout in seconds.",
                }),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("translated_top", "translated_bottom")
    FUNCTION = "translate"
    CATEGORY = "FeiFei"

    def translate(self, text_top, text_bottom, target_lang, temperature,
                  extra_notes="", timeout=120):
        top = (text_top or "").strip()
        bottom = (text_bottom or "").strip()
        if not top and not bottom:
            return ("API Error: both subtitle lines are empty", "")
        api_base, api_key, model = resolve_endpoint("FeiFeiSubtitleTranslator")
        base = (api_base or "").strip().rstrip("/")
        if not base:
            return ("API Error: api_base is empty in config.json", "")
        try:
            seconds = int(timeout)
        except (TypeError, ValueError):
            seconds = 120
        seconds = max(10, min(600, seconds))
        payload = {
            "messages": [
                {"role": "system", "content": TRANSLATE_SYSTEM},
                {"role": "user", "content": build_translate_messages(top, bottom, target_lang, extra_notes)},
            ],
            "temperature": float(temperature),
            "stream": False,
        }
        model_name = (model or "").strip()
        if model_name:
            payload["model"] = model_name
        try:
            res_json = _post_chat_completions(base, payload, timeout=seconds, api_key=api_key)
            choices = res_json.get("choices") if isinstance(res_json, dict) else None
            raw = ""
            if choices and isinstance(choices[0], dict):
                raw = _coerce_text(choices[0].get("message", {}).get("content", ""))
            if not raw:
                raise ValueError("LLM response missing choices[0].message.content")
        except Exception as e:
            return (f"API Error: {e}", "")
        return parse_translate_json(raw)


NODE_CLASS_MAPPINGS = {"FeiFeiSubtitleTranslator": FeiFeiSubtitleTranslator}

NODE_DISPLAY_NAME_MAPPINGS = {"FeiFeiSubtitleTranslator": "Subtitle Translator (API: edit config.json)"}
