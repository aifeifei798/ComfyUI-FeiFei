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
