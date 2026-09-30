import json
import os
import re
import torch
import numpy as np
from PIL import Image
from datetime import datetime

DEFAULT_SUBDIR = "webp_outputs"
# EXIF ImageDescription tag, holds the prompt/seed summary JSON
EXIF_TAG_IMAGE_DESCRIPTION = 270
# WebP's EXIF chunk has a tight size budget, so an oversized summary is
# truncated (the full text stays in the sidecar JSON)
MAX_EXIF_DESC_CHARS = 60000


def _get_output_directory():
    """Lazy-load folder_paths, degrading to ./output in standalone tests"""
    try:
        import folder_paths
        return folder_paths.get_output_directory()
    except Exception:
        fallback = os.path.join(os.getcwd(), "output")
        print(f"[SaveWebP] folder_paths unavailable, falling back to {fallback}")
        return fallback


def _sanitize_subdir(subdir):
    """Block ../ traversal and absolute paths; invalid input falls back to the default"""
    if not isinstance(subdir, str) or not subdir.strip():
        return DEFAULT_SUBDIR
    cleaned = subdir.strip().replace("\\", "/")
    if os.path.isabs(cleaned):
        return DEFAULT_SUBDIR
    parts = [p for p in cleaned.split("/") if p not in ("", ".", "..")]
    # Only safe characters allowed
    safe_parts = [p for p in parts if re.fullmatch(r"[\w\-. ]+", p)]
    if not safe_parts:
        return DEFAULT_SUBDIR
    return os.path.join(*safe_parts)


REDACTED = "<redacted>"
# Widget/input names whose values must never reach a file we write
_SECRET_KEYS = ("api_key", "apikey")
_SECRET_URL_KEYS = ("api_base",)


def _strip_url_userinfo(value):
    """Drop userinfo from a URL so https://user:key@host cannot leak a secret."""
    if not isinstance(value, str) or "@" not in value or "://" not in value:
        return value
    scheme, rest = value.split("://", 1)
    if "/" in rest.split("@", 1)[0] or "?" in rest.split("@", 1)[0]:
        return value  # the @ is in the path or query, not the authority
    return f"{scheme}://{rest.split('@', 1)[1]}"


def _redact_secrets(value, key=None):
    """Recursively replace credential values with a placeholder.

    Anything stored under an api_key-style name is blanked, and a URL stored
    under api_base keeps its host but loses any embedded userinfo. Pure function
    so it can be unit tested without ComfyUI.
    """
    if isinstance(key, str):
        low = key.lower()
        if low in _SECRET_KEYS:
            return REDACTED
        if low in _SECRET_URL_KEYS:
            return _strip_url_userinfo(value)
    if isinstance(value, dict):
        return {k: _redact_secrets(v, k) for k, v in value.items()}
    if isinstance(value, list):
        return [_redact_secrets(v) for v in value]
    return value


def _clip_role_by_links(workflow):
    """Use the workflow links to tell whether each node's output feeds the
    positive or the negative input.

    workflow looks like {"nodes": [{"id":..,"type":..,"inputs":[{"name":..},..]},..],
    "links": [[link_id, from_id, from_slot, to_id, to_slot, type], ..]}.
    Returns {node_id_str: "positive"/"negative"}, or {} on failure.
    """
    roles = {}
    try:
        if isinstance(workflow, str):
            workflow = json.loads(workflow)
        if not isinstance(workflow, dict):
            return {}
        nodes_by_id = {}
        for n in workflow.get("nodes", []) or []:
            if isinstance(n, dict) and "id" in n:
                nodes_by_id[str(n["id"])] = n
        for link in workflow.get("links", []) or []:
            if not isinstance(link, (list, tuple)) or len(link) < 6:
                continue
            _, from_id, _, to_id, to_slot, _ = link[:6]
            to_node = nodes_by_id.get(str(to_id))
            if not isinstance(to_node, dict):
                continue
            inputs = to_node.get("inputs", [])
            name = ""
            if isinstance(inputs, list) and isinstance(to_slot, int) and 0 <= to_slot < len(inputs):
                slot = inputs[to_slot]
                if isinstance(slot, dict):
                    name = str(slot.get("name", ""))
            lname = name.lower()
            if "positive" in lname:
                roles[str(from_id)] = "positive"
            elif "negative" in lname:
                roles[str(from_id)] = "negative"
    except Exception as e:
        print(f"[SaveWebP] workflow link parsing failed, falling back to order: {e}")
    return roles


def _coerce_str(value):
    """External positive/negative validation: anything that is not a string degrades to empty"""
    return value if isinstance(value, str) else ""


def _coerce_seeds(value):
    """External seeds validation: list/tuple items are cast to int one by one; a
    scalar number becomes a single-element list; anything else returns []"""
    if isinstance(value, bool):
        return []
    if isinstance(value, (int, float)):
        return [int(value)]
    if isinstance(value, str):
        s = value.strip()
        try:
            return [int(s)] if s else []
        except ValueError:
            return []
    if isinstance(value, (list, tuple)):
        out = []
        for item in value:
            if isinstance(item, bool):
                continue
            if isinstance(item, (int, float)):
                out.append(int(item))
            elif isinstance(item, str):
                try:
                    out.append(int(item.strip()))
                except ValueError:
                    continue
        return out
    return []


def _extract_summary(prompt, workflow=None):
    """Distill the positive/negative prompts and seeds out of a ComfyUI PROMPT
    dict, tolerating bad input at every step.

    prompt looks like {node_id: {"class_type": ..., "inputs": {...}}}.
    - Text: prefer CLIPTextEncode-family nodes only, with links deciding which
      side is positive. Only when there is no CLIPText node at all does it fall
      back to other text-bearing nodes plus an order heuristic (first is
      positive, second is negative).
    - Seeds: collect every value whose input key is seed / noise_seed.
    """
    seeds = []
    clip_pairs, other_pairs = [], []  # [(node_id_str, text)]
    try:
        roles = _clip_role_by_links(workflow)
        if not isinstance(prompt, dict):
            return {"positive": "", "negative": "", "texts": [], "seeds": []}
        for node_id, node in prompt.items():
            if not isinstance(node, dict):
                continue
            inputs = node.get("inputs", {})
            if not isinstance(inputs, dict):
                continue
            class_type = node.get("class_type", "")
            text = inputs.get("text")
            if isinstance(text, str) and text.strip():
                if "CLIPText" in str(class_type):
                    clip_pairs.append((str(node_id), text))
                else:
                    other_pairs.append((str(node_id), text))
            for key in ("seed", "noise_seed"):
                if key in inputs and isinstance(inputs[key], (int, float)):
                    seeds.append(int(inputs[key]))
    except Exception as e:
        print(f"[SaveWebP] summary extraction failed (image still saved): {e}")
    # Prefer the CLIPText family; only fall back to other text-bearing nodes when
    # there are none, so short text nodes cannot pollute the summary
    pairs = clip_pairs if clip_pairs else other_pairs
    texts_pos, texts_neg, texts_other = [], [], []
    for nid, text in pairs:
        role = roles.get(nid)
        if role == "positive":
            texts_pos.append(text)
        elif role == "negative":
            texts_neg.append(text)
        else:
            texts_other.append(text)
    # Dedupe while preserving order, within each of the three groups
    def _dedup(items):
        seen, uniq = set(), []
        for t in items:
            if t not in seen:
                seen.add(t)
                uniq.append(t)
        return uniq
    texts_pos, texts_neg, texts_other = _dedup(texts_pos), _dedup(texts_neg), _dedup(texts_other)
    # Positive = whatever links called positive, else the first by order; negative
    # works the same way (second entry when links decide nothing)
    positive = texts_pos[0] if texts_pos else (texts_other[0] if texts_other else "")
    if texts_neg:
        negative = texts_neg[0]
    elif not texts_pos and len(texts_other) > 1:
        negative = texts_other[1]
    else:
        negative = ""
    texts = texts_pos + texts_neg + [t for t in texts_other if t not in (positive, negative)]
    seen_seed, uniq_seeds = set(), []
    for s in seeds:
        if s not in seen_seed:
            seen_seed.add(s)
            uniq_seeds.append(s)
    return {
        "positive": positive,
        "negative": negative,
        "texts": texts,
        "seeds": uniq_seeds,
    }


def _build_exif(summary):
    """Summary -> EXIF bytes; returns None on failure (the caller then saves without EXIF)"""
    try:
        from PIL.Image import Exif as PilExif
        desc = json.dumps(summary, ensure_ascii=False, separators=(",", ":"))
        if len(desc) > MAX_EXIF_DESC_CHARS:
            desc = desc[:MAX_EXIF_DESC_CHARS]
        exif = PilExif()
        exif[EXIF_TAG_IMAGE_DESCRIPTION] = desc
        return exif.tobytes()
    except Exception as e:
        print(f"[SaveWebP] EXIF build failed (image still saved): {e}")
        return None

def _read_info_from_image(image_path):
    """Two-path read: prefer the sidecar JSON next to the image (full information),
    fall back to the EXIF ImageDescription.

    Returns (positive, negative, seeds, info_json) with seeds as a list; when
    nothing is found it returns empty strings plus an explanation.
    """
    positive, negative, seeds = "", "", []
    try:
        sidecar_path = os.path.splitext(image_path)[0] + ".json"
        if os.path.isfile(sidecar_path):
            with open(sidecar_path, "r", encoding="utf-8") as f:
                sidecar = json.load(f)
            summary = sidecar.get("summary") or {}
            if not isinstance(summary, dict):
                summary = {}
            positive = _coerce_str(summary.get("positive", ""))
            negative = _coerce_str(summary.get("negative", ""))
            seeds = _coerce_seeds(summary.get("seeds", []))
            return positive, negative, seeds, json.dumps(sidecar, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[LoadWebPInfo] sidecar read failed, trying EXIF: {e}")
    try:
        with Image.open(image_path) as img:
            desc = img.getexif().get(EXIF_TAG_IMAGE_DESCRIPTION, "")
        if desc:
            data = json.loads(desc)
            if not isinstance(data, dict):
                data = {}
            positive = _coerce_str(data.get("positive", ""))
            negative = _coerce_str(data.get("negative", ""))
            seeds = _coerce_seeds(data.get("seeds", []))
            return positive, negative, seeds, json.dumps(data, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[LoadWebPInfo] EXIF read failed: {e}")
    return "", "", [], "No metadata found in sidecar JSON / EXIF (old image or external file?)"


class SaveWebPWithTimestamp:
    def __init__(self):
        self.output_dir = _get_output_directory()
        self.type = "output"
        self.prefix_append = ""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "quality": ("INT", {"default": 90, "min": 1, "max": 100, "step": 1}),
                "lossless": ("BOOLEAN", {"default": False}),
                "subdir": ("STRING", {"default": "webp_outputs"}),
                "embed_metadata": ("BOOLEAN", {"default": True}),
                "save_json": ("BOOLEAN", {"default": True}),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "save_images"
    OUTPUT_NODE = True
    CATEGORY = "FeiFei"

    def save_images(self, images, quality, lossless, subdir,
                    embed_metadata=True, save_json=True,
                    prompt=None, extra_pnginfo=None):
        # Resolve the save path (traversal-safe)
        safe_subdir = _sanitize_subdir(subdir)
        full_output_folder = os.path.join(self.output_dir, safe_subdir)
        os.makedirs(full_output_folder, exist_ok=True)

        results = list()
        # One timestamp prefix plus an index per batch, so several images saved in
        # the same millisecond cannot overwrite each other
        batch_stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]

        # Prompt/seed summary (a failure here must not block saving; the workflow is
        # what lets links trace which side is positive)
        workflow = extra_pnginfo.get("workflow") if isinstance(extra_pnginfo, dict) else None
        summary = _extract_summary(prompt, workflow) if embed_metadata or save_json else None
        exif_bytes = _build_exif({
            "node": "SaveWebPWithTimestamp",
            "created_at": batch_stamp,
            "positive": (summary or {}).get("positive", ""),
            "negative": (summary or {}).get("negative", ""),
            "seeds": (summary or {}).get("seeds", []),
        }) if embed_metadata else None

        for idx, image in enumerate(images):
            # Convert the tensor to a PIL Image
            i = 255. * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            if img.mode not in ("RGB", "RGBA"):
                img = img.convert("RGB")

            # Build the file name: YYYYMMDD_HHMMSS_millis_index.webp
            file_name = f"{batch_stamp}_{idx:03d}.webp"
            file_path = os.path.join(full_output_folder, file_name)

            # Save as WebP (quality is not passed in lossless mode, which Pillow
            # would warn about and then ignore)
            save_kwargs = {"format": "WEBP"}
            if exif_bytes is not None:
                save_kwargs["exif"] = exif_bytes
            if lossless:
                img.save(file_path, lossless=True, **save_kwargs)
            else:
                img.save(file_path, quality=int(quality), lossless=False, **save_kwargs)

            # Sidecar JSON: summary plus the full prompt/workflow
            if save_json:
                try:
                    sidecar = {
                        "file": file_name,
                        "created_at": batch_stamp,
                        "summary": summary,
                        # The prompt dict is keyed by input name, so credentials in
                        # it can be scrubbed. The workflow copy is not: ComfyUI
                        # stores widget values as a positional array with no names.
                        "prompt": _redact_secrets(prompt),
                        "workflow": workflow,
                    }
                    with open(os.path.splitext(file_path)[0] + ".json", "w", encoding="utf-8") as f:
                        json.dump(sidecar, f, ensure_ascii=False, indent=2)
                except Exception as e:
                    print(f"[SaveWebP] sidecar JSON write failed (image still saved): {e}")

            results.append({
                "filename": file_name,
                "subfolder": safe_subdir,
                "type": self.type
            })

        return {"ui": {"images": results}}


class LoadWebPInfo:
    """Read back the prompts/seeds written by this pack's SaveWebP node (sidecar JSON first, EXIF fallback)"""

    @classmethod
    def INPUT_TYPES(s):
        try:
            import folder_paths
            input_dir = folder_paths.get_input_directory()
            files = [f for f in os.listdir(input_dir)
                     if os.path.isfile(os.path.join(input_dir, f))]
            files = folder_paths.filter_files_content_types(files, ["image"])
        except Exception:
            files = []
        return {
            "required": {
                "image": (sorted(files), {"image_upload": True}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("positive", "negative", "seeds", "info_json")
    FUNCTION = "load_info"
    CATEGORY = "FeiFei"

    def load_info(self, image):
        try:
            import folder_paths
            image_path = folder_paths.get_annotated_filepath(image)
        except Exception:
            image_path = image if os.path.isfile(image) else None
        if not image_path or not os.path.isfile(image_path):
            return ("", "", "", f"Image file not found: {image}")
        positive, negative, seeds, info_json = _read_info_from_image(image_path)
        if not isinstance(seeds, (list, tuple)):
            seeds = _coerce_seeds(seeds)
        seeds_str = ", ".join(str(s) for s in seeds)
        return (_coerce_str(positive), _coerce_str(negative), seeds_str, info_json)


NODE_CLASS_MAPPINGS = {
    "SaveWebPWithTimestamp": SaveWebPWithTimestamp,
    "LoadWebPInfo": LoadWebPInfo,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SaveWebPWithTimestamp": "Save WebP (Timestamp)",
    "LoadWebPInfo": "Load WebP Info",
}
