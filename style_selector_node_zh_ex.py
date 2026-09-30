# -----------------------------------------------------------------
# Style Selector EX (node logic; template data lives in style_data.py / juese_data.py)
# -----------------------------------------------------------------
import random
import time
import re

from .style_data import style_list, resolve_style_name
from .juese_data import juese_list, resolve_juese_name


def _dedupe_names(entries):
    """Dedupe by first appearance. A duplicate name only ever matches its first
    entry on lookup, so keeping it in the dropdown is just noise (the old data
    had duplicates to begin with)."""
    seen = set()
    out = []
    for item in entries:
        name = item.get("name")
        if not name or name in seen:
            continue
        seen.add(name)
        out.append(name)
    return out


class StyleSelectorNodeZhex:
    """
    Repaired custom node
    """

    # Give ComfyUI one legal placeholder entry when a list is empty, so an empty
    # dropdown cannot raise
    style_names = _dedupe_names([s for s in style_list if isinstance(s, dict)]) or ["(None)"]
    juese_names = _dedupe_names([j for j in juese_list if isinstance(j, dict)]) or ["(None)"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt1": (
                    "STRING",
                    {"multiline": True, "default": "A cute Japanese idol girl, portrait, masterpiece, best quality"},
                ),
                "prompt2": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "Underwater photography style (shooting through glass), twin tail girl submerged in water, cheeks puffed out blowing bubbles, twin tails floating freely in the water, dreamy and soft, cute expression.",
                    },
                ),
                "prompt3": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "Japanese gravure idol photography, realistic human, Fujifilm Provia color grading, soft pastel tones, high key lighting, clear skin texture, cinematic bokeh, 8k, highly detailed, natural skin, gorgeous, sharp focus, masterpiece, best quality.",
                    },
                ),
                # style_names/juese_names are guaranteed non-empty (see the class
                # attributes); an empty list makes ComfyUI raise
                "style_name": (cls.style_names,),
                "juese_names": (cls.juese_names,),
                "random_style": ("BOOLEAN", {"default": False}),
            },
            # Prompts 4-7: pure input sockets (forceInput takes no textbox), meant
            # for upstream nodes such as Prompt Director's positive_prompt; None
            # when not connected
            "optional": {
                "prompt4": ("STRING", {"forceInput": True}),
                "prompt5": ("STRING", {"forceInput": True}),
                "prompt6": ("STRING", {"forceInput": True}),
                "prompt7": ("STRING", {"forceInput": True}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("positive_prompt", "negative_prompt")
    FUNCTION = "apply_style"
    CATEGORY = "FeiFei"

    @classmethod
    def IS_CHANGED(
        cls, prompt1, prompt2, prompt3, style_name, juese_names, random_style,
        prompt4=None, prompt5=None, prompt6=None, prompt7=None,
    ):
        # Force a refresh while random is on; otherwise return None and let ComfyUI
        # cache on the input hash. Note: returning float("NaN") here would always
        # look dirty (NaN != NaN) and the cache would never hit.
        if random_style:
            return time.time_ns()
        return None

    def apply_style(
        self, prompt1, prompt2, prompt3, style_name, juese_names, random_style,
        prompt4=None, prompt5=None, prompt6=None, prompt7=None,
    ):
        # 1. Join the base prompts. The space stops words from running together;
        # prompt4~7 (external) are appended after the three textboxes in order.
        base_parts = [prompt1, prompt2, prompt3, prompt4, prompt5, prompt6, prompt7]
        prompt = " ".join(
            s for s in (str(p).strip() for p in base_parts if p is not None) if s
        ).strip()

        # Seed the negative prompt
        current_negative = ""
        # ===========================
        # Step 1: character (Juese)
        # Find the character, then append its prompt after your input prompt
        # ===========================
        # Template names are already English; Chinese/Japanese names stored in old
        # workflows resolve through the alias table before lookup
        selected_juese = next(
            (j for j in juese_list if j["name"] == resolve_juese_name(juese_names)), None
        )

        if (
            selected_juese and selected_juese["name"] != "(None)"
        ):  # assuming you have an option named (None)
            # Fetch the character prompt, empty when there is none
            juese_prompt = selected_juese.get("prompt", "")
            juese_negative = selected_juese.get("negative_prompt", "")

            # Append the character prompt to the main prompt
            prompt = f"{prompt}, {juese_prompt}"
            current_negative = f"{current_negative}, {juese_negative}"
        # ===========================
        # Step 2: style
        # Substitute the character-processed prompt into the style template
        # ===========================
        resolved_style_name = resolve_style_name(style_name)
        selected_style = None
        if random_style:
            # Drop invalid styles
            eligible_styles = [s for s in style_list if s["name"] != "(None)"]
            if eligible_styles:
                selected_style = random.choice(eligible_styles)
                print(
                    f"[StyleSelector] Random style selected: {selected_style['name']}"
                )
            else:
                # Nothing selectable, fall back to the current choice
                selected_style = next(
                    (s for s in style_list if s["name"] == resolved_style_name), None
                )
        else:
            selected_style = next(
                (s for s in style_list if s["name"] == resolved_style_name), None
            )
        # Apply the style template
        if selected_style and selected_style["name"] != "(None)":
            prompt_template = selected_style.get("prompt", "{prompt}")
            negative_template = selected_style.get("negative_prompt", "")
            # Substitute {prompt} in the template
            final_positive = prompt_template.replace("{prompt}", prompt)

            # Handle the negative prompt. A style's negative prompt is normally
            # fixed, or appended to. If it happens to carry {prompt} too (rare)
            # substitute, otherwise just concatenate.
            if "{prompt}" in negative_template:
                final_negative = negative_template.replace("{prompt}", current_negative)
            else:
                # No {prompt} in the template usually means the style ships generic
                # negative terms, so tack the character's negative prompt onto it
                final_negative = f"{negative_template}, {current_negative}".strip().strip(",").strip()
        else:
            # No style selected, output the current result as-is
            final_positive = prompt
            final_negative = current_negative
        # Clean up redundant commas and spaces
        final_positive = final_positive.strip().strip(",").strip()
        final_positive = re.sub(r",\s*,", ",", final_positive)
        final_positive = re.sub(r"\s+", " ", final_positive)
        final_positive = (
            final_positive.replace(", ,", ",").replace(",,", ",").strip().strip(",").strip()
        )
        final_negative = final_negative.strip().strip(",").strip()
        return (final_positive, final_negative)


# -----------------------------------------------------------------
#  Mapping dicts ComfyUI requires
#  These tell ComfyUI how to load and display the node
# -----------------------------------------------------------------
NODE_CLASS_MAPPINGS = {"StyleSelectorNodeZhex": StyleSelectorNodeZhex}
NODE_DISPLAY_NAME_MAPPINGS = {"StyleSelectorNodeZhex": "Style Selector EX"}
