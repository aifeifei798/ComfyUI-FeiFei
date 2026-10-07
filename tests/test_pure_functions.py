"""Standalone unit tests for ComfyUI-FeiFei pure helpers.

Run from the pack root with the ComfyUI venv (torch available) or plain
python3 (torch-dependent cases skip gracefully):

    /home/feifei/mnt/nvme_1t/develop/ComfyUI/.venv/bin/python -m unittest discover -s tests -v
    python3 -m unittest discover -s tests -v
"""

import importlib
import json
import os
import sys
import types
import unittest

PACK_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Node files use relative imports (from .llm_common import ...), and the pack
# directory name contains a hyphen, so expose it under a synthetic package
# name and import submodules through it.
_PKG_NAME = "feifei_pack"
_pkg = types.ModuleType(_PKG_NAME)
_pkg.__path__ = [PACK_ROOT]
sys.modules[_PKG_NAME] = _pkg


def _mod(name):
    return importlib.import_module(f"{_PKG_NAME}.{name}")


def _torch_available():
    try:
        import torch  # noqa: F401
        return True
    except Exception:
        return False


_llm_common = _mod("llm_common")
_coerce_text = _llm_common._coerce_text
_extract_json_object = _llm_common._extract_json_object
_normalize_base = _llm_common._normalize_base
_strip_code_fences = _llm_common._strip_code_fences
_strip_v1 = _llm_common._strip_v1

_aspect = _mod("aspect_ratio_node")
_align16 = _aspect._align16
_parse_ratio = _aspect._parse_ratio
calc_size = _aspect.calc_size

_director = _mod("prompt_director_node")
build_user_message = _director.build_user_message
clamp_aspect_ratio = _director.clamp_aspect_ratio
parse_director_json = _director.parse_director_json

_qwen = _mod("qwen_prompt_node")
parse_wh_ratio = _qwen.parse_wh_ratio

_webp = _mod("ComfyUI_SaveWebP.save_webp_node")
_coerce_seeds = _webp._coerce_seeds
_extract_summary = _webp._extract_summary
_redact_secrets = _webp._redact_secrets
_sanitize_subdir = _webp._sanitize_subdir


class TestLlmCommon(unittest.TestCase):
    def test_normalize_base(self):
        self.assertEqual(_normalize_base("http://127.0.0.1:8080"), "http://127.0.0.1:8080/v1")
        self.assertEqual(_normalize_base("http://127.0.0.1:8080/"), "http://127.0.0.1:8080/v1")
        self.assertEqual(_normalize_base("http://127.0.0.1:8080/v1"), "http://127.0.0.1:8080/v1")
        self.assertEqual(_normalize_base("http://127.0.0.1:8080/v1/"), "http://127.0.0.1:8080/v1")
        self.assertEqual(_normalize_base(""), "")

    def test_strip_v1(self):
        self.assertEqual(_strip_v1("http://127.0.0.1:8080/v1"), "http://127.0.0.1:8080")
        self.assertEqual(_strip_v1("http://127.0.0.1:8080"), "http://127.0.0.1:8080")

    def test_strip_code_fences(self):
        self.assertEqual(_strip_code_fences('```json\n{"a": 1}\n```'), '{"a": 1}')
        self.assertEqual(_strip_code_fences('```\n{"a": 1}\n```'), '{"a": 1}')
        self.assertEqual(_strip_code_fences('{"a": 1}'), '{"a": 1}')
        self.assertEqual(_strip_code_fences(None), "")

    def test_extract_json_object(self):
        self.assertEqual(_extract_json_object('{"a": 1}'), {"a": 1})
        self.assertEqual(
            _extract_json_object('prefix {"a": 1} suffix'), {"a": 1}
        )
        self.assertEqual(
            _extract_json_object('```json\n{"positive_prompt": "x"}\n```'),
            {"positive_prompt": "x"},
        )
        self.assertIsNone(_extract_json_object("no json here"))
        self.assertIsNone(_extract_json_object(""))

    def test_coerce_text(self):
        self.assertEqual(_coerce_text("  hi  "), "hi")
        self.assertEqual(
            _coerce_text([{"type": "text", "text": "a"}, "b"]), "a\nb"
        )
        self.assertEqual(_coerce_text(None), "")


class TestAspectRatio(unittest.TestCase):
    def test_parse_ratio(self):
        self.assertEqual(_parse_ratio("16:9"), (16.0, 9.0))
        self.assertEqual(_parse_ratio("2.39:1"), (2.39, 1.0))
        self.assertEqual(_parse_ratio("bogus"), (1, 1))
        self.assertEqual(_parse_ratio(None), (1, 1))

    def test_align16(self):
        self.assertEqual(_align16(1024), 1024)
        self.assertEqual(_align16(1000) % 16, 0)
        self.assertGreaterEqual(_align16(0), 16)

    def test_calc_size_16_aligned(self):
        for ratio in ("1:1", "16:9", "9:16", "21:9", "2.39:1"):
            for lock in ("Short Side (recommended)", "Fixed Width (width=base)", "Fixed Height (height=base)"):
                w, h = calc_size(ratio, lock, 1024)
                self.assertEqual(w % 16, 0, f"{ratio}/{lock}")
                self.assertEqual(h % 16, 0, f"{ratio}/{lock}")

    def test_calc_size_known(self):
        w, h = calc_size("1:1", "Short Side (recommended)", 1024)
        self.assertEqual((w, h), (1024, 1024))
        w, h = calc_size("16:9", "Fixed Height (height=base)", 1024)
        self.assertEqual(h, 1024)
        self.assertGreater(w, h)


class TestPromptDirector(unittest.TestCase):
    def test_clamp(self):
        self.assertEqual(clamp_aspect_ratio("16:9"), "16:9")
        self.assertEqual(clamp_aspect_ratio("16.0:9.0"), "16:9")
        self.assertEqual(clamp_aspect_ratio("bogus"), "1:1")
        self.assertEqual(clamp_aspect_ratio(None), "1:1")

    def test_parse_director_json_ok(self):
        raw = json.dumps({
            "positive_prompt": "a cat, cinematic light",
            "negative_prompt": "blurry",
            "aspect_ratio": "16:9",
        })
        pos, neg, ratio = parse_director_json(raw)
        self.assertIn("cat", pos)
        self.assertEqual(neg, "blurry")
        self.assertEqual(ratio, "16:9")

    def test_parse_director_json_bad(self):
        pos, neg, ratio = parse_director_json("not json at all")
        self.assertTrue(pos.startswith("API Error"))
        self.assertEqual(ratio, "1:1")

    def test_build_user_message(self):
        msg = build_user_message("a cat", "Flux")
        self.assertIn("a cat", msg)
        self.assertIn("Flux", msg)


class TestParseWhRatio(unittest.TestCase):
    def test_known_map(self):
        w, h = parse_wh_ratio("1:1")
        self.assertEqual((w, h), (1536, 1536))

    def test_custom_aligned(self):
        w, h = parse_wh_ratio("16:9")
        self.assertEqual(w % 16, 0)
        self.assertEqual(h % 16, 0)

    def test_bogus_falls_back(self):
        self.assertEqual(parse_wh_ratio("100:1")[0] % 16, 0)
        self.assertEqual(parse_wh_ratio(""), (1536, 1536))
        self.assertEqual(parse_wh_ratio(None), (1536, 1536))


class TestSaveWebP(unittest.TestCase):
    def test_sanitize_subdir(self):
        self.assertEqual(_sanitize_subdir("../../etc"), "etc")
        self.assertEqual(_sanitize_subdir("/abs/path"), "webp_outputs")
        self.assertEqual(_sanitize_subdir(""), "webp_outputs")
        self.assertEqual(_sanitize_subdir("a/b"), os.path.join("a", "b"))

    def test_redact_secrets(self):
        self.assertEqual(
            _redact_secrets({"api_key": "sk-123", "other": 1}),
            {"api_key": "<redacted>", "other": 1},
        )
        self.assertEqual(
            _redact_secrets({"api_base": "https://user:pass@host/v1"}),
            {"api_base": "https://host/v1"},
        )

    def test_coerce_seeds(self):
        self.assertEqual(_coerce_seeds(42), [42])
        self.assertEqual(_coerce_seeds(True), [])
        self.assertEqual(_coerce_seeds([1, "2", "x", True]), [1, 2])

    def test_extract_summary_tolerates_garbage(self):
        out = _extract_summary(None, None)
        self.assertEqual(out["positive"], "")
        self.assertEqual(out["seeds"], [])
        out = _extract_summary(
            {"1": {"class_type": "CLIPTextEncode", "inputs": {"text": "a cat", "seed": 7}}},
            None,
        )
        self.assertEqual(out["positive"], "a cat")
        self.assertEqual(out["seeds"], [7])


class TestStyleDedupe(unittest.TestCase):
    def test_dedupe_names(self):
        _dedupe_names = _mod("style_selector_node_zh_ex")._dedupe_names

        entries = [{"name": "a"}, {"name": "a"}, {"name": "b"}, {"no": 1}]
        self.assertEqual(_dedupe_names(entries), ["a", "b"])


@unittest.skipUnless(_torch_available(), "torch not installed")
class TestTorchNodes(unittest.TestCase):
    def test_film_resolve_params_modes(self):
        _resolve_params = _mod("film_grain_node")._resolve_params

        sliders = {
            "grain_amount": 0.5, "grain_size": 1.0, "grain_shadows": 0.5,
            "grain_chroma": 0.2, "halation": 0.2, "halation_threshold": 0.8,
            "halation_radius": 1.0, "vignette": 0.1, "vignette_size": 0.75,
            "tone": 0.2, "micro_contrast": 0.1, "chroma_shift": 0.8,
            "split_tone": 0.05,
        }
        custom = _resolve_params(None, 1.0, sliders)
        self.assertAlmostEqual(custom["grain_amount"], 0.5)
        preset = _resolve_params("Portra 400", 0.0, sliders)
        self.assertAlmostEqual(preset["grain_amount"], 0.0)
        half = _resolve_params("Portra 400", 0.5, sliders)
        full = _resolve_params("Portra 400", 1.0, sliders)
        self.assertAlmostEqual(half["grain_amount"], full["grain_amount"] * 0.5)

    def test_film_finish_shape(self):
        import torch

        _film = _mod("film_grain_node")
        _resolve_params, film_finish = _film._resolve_params, _film.film_finish

        frame = torch.rand(64, 64, 3)
        params = _resolve_params(
            "Digital Clean", 1.0,
            {"grain_amount": 0.1, "grain_size": 1.0, "grain_shadows": 0.5,
             "grain_chroma": 0.2, "halation": 0.0, "halation_threshold": 0.8,
             "halation_radius": 1.0, "vignette": 0.0, "vignette_size": 0.75,
             "tone": 0.1, "micro_contrast": 0.0, "chroma_shift": 0.0,
             "split_tone": 0.0},
        )
        out = film_finish(frame, params, 0)
        self.assertEqual(tuple(out.shape), (64, 64, 3))

    def test_cinematic_frame_size(self):
        from PIL import Image

        cinematic_frame = _mod("cinematic_frame_node").cinematic_frame

        img = Image.new("RGB", (512, 512), (200, 100, 50))
        out = cinematic_frame(img, aspect_ratio="16:9", base_side=512,
                              subtitle_top="", subtitle_bottom="")
        self.assertEqual(out.size[1], 512)


if __name__ == "__main__":
    unittest.main()
