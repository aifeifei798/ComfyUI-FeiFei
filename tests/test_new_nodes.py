"""Tests for the new pipeline nodes (no network, torch-gated where needed)."""

import os
import sys
import types
import importlib
import unittest

PACK_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_PKG_NAME = "feifei_pack"
if _PKG_NAME not in sys.modules:
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


class TestNegativeLibrary(unittest.TestCase):
    def test_build_negative(self):
        build_negative = _mod("negative_library_node").build_negative
        out = build_negative("SDXL Base", "", None)
        self.assertIn("watermark", out)
        self.assertEqual(build_negative("(None)", "", None), "")
        out = build_negative("(None)", "blurry, blurry", "text")
        self.assertEqual(out.count("blurry"), 1)
        self.assertIn("text", out)


class TestSubtitleTranslator(unittest.TestCase):
    def test_build_and_parse(self):
        mod = _mod("subtitle_translator_node")
        msg = mod.build_translate_messages("Hello", "", "Chinese")
        self.assertIn("Chinese", msg)
        top, bottom = mod.parse_translate_json('{"top": "你好", "bottom": ""}')
        self.assertEqual(top, "你好")
        self.assertEqual(bottom, "")
        top, bottom = mod.parse_translate_json("garbage")
        self.assertTrue(top.startswith("API Error"))


class TestLocalSourceHints(unittest.TestCase):
    """Every LLM node returns the download command when its folder is missing."""

    def setUp(self):
        # The folder comes from config.json, so point the resolver at a missing one
        self.llm = _mod("llm_common")
        self.original = self.llm.resolve_local_model
        self.llm.resolve_local_model = lambda node_class="": (
            "LiquidAI/LFM3-VL-2B", "/nonexistent/feifei-model")

    def tearDown(self):
        self.llm.resolve_local_model = self.original

    def test_prompt_enhancer(self):
        llm = _mod("llm_common")
        out = _mod("qwen_prompt_node").QwenImagePromptEnhancer().enhance_prompt(
            source=llm.SOURCE_LOCAL, user_prompt="a cat", mode="T2I", temperature=0.7)
        self.assertIn("hf download LiquidAI/LFM3-VL-2B", out[0])
        self.assertIn("/nonexistent/feifei-model", out[0])
        self.assertEqual(out[1], "")
        self.assertEqual(out[5], "")

    def test_prompt_director(self):
        llm = _mod("llm_common")
        out = _mod("prompt_director_node").FeiFeiPromptDirector().direct(
            source=llm.SOURCE_LOCAL, keywords="cyberpunk, rain", model_style="Flux",
            temperature=0.7, thinking_mode=llm.THINKING_OURS)
        self.assertIn("hf download LiquidAI/LFM3-VL-2B", out[0])
        self.assertEqual(out[2], "1:1")

    def test_subtitle_translator(self):
        llm = _mod("llm_common")
        out = _mod("subtitle_translator_node").FeiFeiSubtitleTranslator().translate(
            source=llm.SOURCE_LOCAL, text_top="hello", text_bottom="", target_lang="Chinese",
            temperature=0.3)
        self.assertIn("hf download LiquidAI/LFM3-VL-2B", out[0])

    def test_image_nodes(self):
        import tempfile

        try:
            from PIL import Image
        except ImportError:
            self.skipTest("Pillow not installed")
        llm = _mod("llm_common")

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "frame.png")
            Image.new("RGB", (32, 32), (10, 20, 30)).save(path)

            out = _mod("image_describe_node").FeiFeiImageDescribe().describe_image(
                source=llm.SOURCE_LOCAL, image=path, instruction="", temperature=0.3,
                max_tokens=64, max_side=256)
            self.assertIn("hf download LiquidAI/LFM3-VL-2B", out[0])

            out = _mod("image_caption_node").FeiFeiImageCaptioner().caption_image(
                source=llm.SOURCE_LOCAL, image=path, instruction="", temperature=0.7,
                max_tokens=64, max_side=256, thinking_mode=llm.THINKING_OURS)
            self.assertIn("hf download LiquidAI/LFM3-VL-2B", out[0])
            self.assertEqual(out[1], "")


@unittest.skipUnless(_torch_available(), "torch not installed")
class TestCompareAndSafeArea(unittest.TestCase):
    def test_compare_shapes(self):
        import torch
        from PIL import Image

        compare_pils = _mod("compare_node").compare_pils
        before = Image.new("RGB", (64, 48), (200, 30, 30))
        after = Image.new("RGB", (32, 32), (30, 30, 200))
        side = compare_pils(before, after, mode="Side by Side (H)",
                            show_labels=False)
        self.assertEqual(side.size, (128, 48))
        stack = compare_pils(before, after, mode="Stack (V)",
                             show_labels=False)
        self.assertEqual(stack.size, (64, 96))
        wipe = compare_pils(before, after, mode="Wipe (Left-Right)",
                            divider=0.25, show_labels=True)
        self.assertEqual(wipe.size, (64, 48))

    def test_compare_node_batch(self):
        import torch

        cls = _mod("compare_node").FeiFeiBeforeAfterCompare
        node = cls()
        b = torch.rand(2, 32, 48, 3)
        a = torch.rand(1, 32, 48, 3)
        out, = node.compare(b, a, "Side by Side (H)", 0.5, 2, False,
                            "Before", "After")
        self.assertEqual(tuple(out.shape), (2, 32, 96, 3))

    def test_safe_area_shape(self):
        import torch
        from PIL import Image

        draw_safe_area = _mod("safe_area_node").draw_safe_area
        img = Image.new("RGB", (64, 48), (100, 100, 100))
        out = draw_safe_area(img, safe_margin=0.1)
        self.assertEqual(out.size, (64, 48))
        cls = _mod("safe_area_node").FeiFeiSafeAreaOverlay
        node = cls()
        tensor = torch.rand(1, 48, 64, 3)
        result, = node.overlay(tensor, 0.1, True, True, 2)
        self.assertEqual(tuple(result.shape), (1, 48, 64, 3))


if __name__ == "__main__":
    unittest.main()
