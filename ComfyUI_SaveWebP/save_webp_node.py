import json
import os
import re
import torch
import numpy as np
from PIL import Image
from datetime import datetime

DEFAULT_SUBDIR = "webp_outputs"
# EXIF ImageDescription  tag，存提示词/种子摘要 JSON
EXIF_TAG_IMAGE_DESCRIPTION = 270
# WebP EXIF chunk 体积限制较严，摘要超限则截断（全文保留在 sidecar JSON）
MAX_EXIF_DESC_CHARS = 60000


def _get_output_directory():
    """懒加载 folder_paths， standalone 测试时降级到 ./output"""
    try:
        import folder_paths
        return folder_paths.get_output_directory()
    except Exception:
        fallback = os.path.join(os.getcwd(), "output")
        print(f"[SaveWebP] folder_paths unavailable, falling back to {fallback}")
        return fallback


def _sanitize_subdir(subdir):
    """防止 ../ 目录穿越与绝对路径，非法输入回退默认值"""
    if not isinstance(subdir, str) or not subdir.strip():
        return DEFAULT_SUBDIR
    cleaned = subdir.strip().replace("\\", "/")
    if os.path.isabs(cleaned):
        return DEFAULT_SUBDIR
    parts = [p for p in cleaned.split("/") if p not in ("", ".", "..")]
    # 仅允许 安全字符
    safe_parts = [p for p in parts if re.fullmatch(r"[\w\-. ]+", p)]
    if not safe_parts:
        return DEFAULT_SUBDIR
    return os.path.join(*safe_parts)


def _extract_summary(prompt):
    """从 ComfyUI PROMPT dict 提炼正负提示词与种子，全程容错。

    prompt 形如 {node_id: {"class_type": ..., "inputs": {...}}}。
    - 文本：收集 CLIPTextEncode(及同类)节点的 text 输入
    - 种子：收集所有 inputs 中 key 为 seed / noise_seed 的值
    """
    texts = []
    seeds = []
    try:
        if not isinstance(prompt, dict):
            return {"positive": "", "negative": "", "texts": [], "seeds": []}
        for node in prompt.values():
            if not isinstance(node, dict):
                continue
            inputs = node.get("inputs", {})
            if not isinstance(inputs, dict):
                continue
            class_type = node.get("class_type", "")
            text = inputs.get("text")
            if isinstance(text, str) and text.strip():
                # CLIPTextEncode 系节点优先；其他带 text 的节点也收录
                if "CLIPText" in str(class_type):
                    texts.append(text)
                elif len(text) < 20000:
                    texts.append(text)
            for key in ("seed", "noise_seed"):
                if key in inputs and isinstance(inputs[key], (int, float)):
                    seeds.append(int(inputs[key]))
    except Exception as e:
        print(f"[SaveWebP] summary extraction failed (image still saved): {e}")
    # 去重保序
    seen_text, uniq_texts = set(), []
    for t in texts:
        if t not in seen_text:
            seen_text.add(t)
            uniq_texts.append(t)
    seen_seed, uniq_seeds = set(), []
    for s in seeds:
        if s not in seen_seed:
            seen_seed.add(s)
            uniq_seeds.append(s)
    # 启发式：第一个文本为正、第二个为负（与官方 save 逻辑无关，仅方便阅读）
    return {
        "positive": uniq_texts[0] if len(uniq_texts) > 0 else "",
        "negative": uniq_texts[1] if len(uniq_texts) > 1 else "",
        "texts": uniq_texts,
        "seeds": uniq_seeds,
    }


def _build_exif(summary):
    """摘要 -> EXIF 字节；失败返回 None（调用方直接不带 exif 保存）"""
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
    """双路读取：优先同目录 sidecar JSON（信息全），回退 EXIF ImageDescription。

    返回 (positive, negative, seeds_str, info_json)，找不到返回空字符串+说明。
    """
    positive, negative, seeds = "", "", []
    try:
        sidecar_path = os.path.splitext(image_path)[0] + ".json"
        if os.path.isfile(sidecar_path):
            with open(sidecar_path, "r", encoding="utf-8") as f:
                sidecar = json.load(f)
            summary = sidecar.get("summary") or {}
            positive = summary.get("positive", "") or ""
            negative = summary.get("negative", "") or ""
            seeds = summary.get("seeds", []) or []
            return positive, negative, seeds, json.dumps(sidecar, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[LoadWebPInfo] sidecar read failed, trying EXIF: {e}")
    try:
        with Image.open(image_path) as img:
            desc = img.getexif().get(EXIF_TAG_IMAGE_DESCRIPTION, "")
        if desc:
            data = json.loads(desc)
            positive = data.get("positive", "") or ""
            negative = data.get("negative", "") or ""
            seeds = data.get("seeds", []) or []
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
        # 确定保存路径（防穿越）
        safe_subdir = _sanitize_subdir(subdir)
        full_output_folder = os.path.join(self.output_dir, safe_subdir)
        os.makedirs(full_output_folder, exist_ok=True)

        results = list()
        # 同一批次用同一时间戳前缀 + 序号，保证毫秒内多图不覆盖
        batch_stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]

        # 提示词/种子摘要（失败不阻断存图）
        summary = _extract_summary(prompt) if embed_metadata or save_json else None
        exif_bytes = _build_exif({
            "node": "SaveWebPWithTimestamp",
            "created_at": batch_stamp,
            "positive": (summary or {}).get("positive", ""),
            "negative": (summary or {}).get("negative", ""),
            "seeds": (summary or {}).get("seeds", []),
        }) if embed_metadata else None

        for idx, image in enumerate(images):
            # 将张量转换为 PIL Image
            i = 255. * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            if img.mode not in ("RGB", "RGBA"):
                img = img.convert("RGB")

            # 生成文件名: YYYYMMDD_HHMMSS_毫秒_序号.webp
            file_name = f"{batch_stamp}_{idx:03d}.webp"
            file_path = os.path.join(full_output_folder, file_name)

            # 保存为 WebP（lossless 时不传 quality，避免 Pillow 告警/忽略）
            save_kwargs = {"format": "WEBP"}
            if exif_bytes is not None:
                save_kwargs["exif"] = exif_bytes
            if lossless:
                img.save(file_path, lossless=True, **save_kwargs)
            else:
                img.save(file_path, quality=int(quality), lossless=False, **save_kwargs)

            # sidecar JSON：摘要 + 完整 prompt/workflow
            if save_json:
                try:
                    sidecar = {
                        "file": file_name,
                        "created_at": batch_stamp,
                        "summary": summary,
                        "prompt": prompt,
                        "workflow": extra_pnginfo.get("workflow") if isinstance(extra_pnginfo, dict) else extra_pnginfo,
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
    """读取本包 SaveWebP 节点存入的提示词/种子（sidecar JSON 优先，EXIF 兜底）"""

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
        seeds_str = ", ".join(str(s) for s in seeds)
        return (positive, negative, seeds_str, info_json)


NODE_CLASS_MAPPINGS = {
    "SaveWebPWithTimestamp": SaveWebPWithTimestamp,
    "LoadWebPInfo": LoadWebPInfo,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SaveWebPWithTimestamp": "Save WebP (Timestamp)",
    "LoadWebPInfo": "Load WebP Info",
}
