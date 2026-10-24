"""Unit tests for the Harmonic Dynamic EQ node: pure math + sampler-CFG contract,
no ComfyUI required. Skips gracefully when torch is missing."""

import importlib
import os
import sys
import types
import unittest

PACK_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_PKG_NAME = "feifei_pack"
if _PKG_NAME not in sys.modules:
    _pkg = types.ModuleType(_PKG_NAME)
    _pkg.__path__ = [PACK_ROOT]
    sys.modules[_PKG_NAME] = _pkg

try:
    import torch
except Exception:
    torch = None

_harmonic = None
if torch is not None:
    _harmonic = importlib.import_module(f"{_PKG_NAME}.harmonic_eq_node")


@unittest.skipUnless(torch is not None, "torch not available")
class TestHarmonicDynamicEqMath(unittest.TestCase):

    def test_dtype_and_shape_preserved(self):
        eq = _harmonic.harmonic_dynamic_eq
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            for shape in ((2, 128, 640), (2, 4, 64, 64), (1, 8, 3, 16, 16), (2, 32)):
                with self.subTest(dtype=dtype, shape=shape):
                    x = (torch.randn(*shape) * 0.7).to(dtype)
                    y = eq(x, 24.0, 0.5, 0.2, 0.2)
                    self.assertEqual(y.dtype, dtype)
                    self.assertEqual(y.shape, x.shape)
                    self.assertTrue(torch.isfinite(y.float()).all())

    def test_bit_exact_transparent(self):
        eq = _harmonic.harmonic_dynamic_eq
        x = torch.randn(2, 77, 640)
        self.assertTrue(torch.equal(x, eq(x, 24.0, 0.0, 0.2, 0.0)))
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype):
                band = torch.full((2, 100, 32), 0.5, dtype=dtype)
                self.assertTrue(torch.equal(band, eq(band, 24.0, 0.5, 0.2, 0.2)))

    def test_peak_compression_matches_curve(self):
        eq = _harmonic.harmonic_dynamic_eq
        x = torch.zeros(1, 16, 8)
        x[0, 0, 0] = 1000.0
        y = eq(x, 24.0, 0.5, 0.2, 0.2)
        gain = y[0, 0, 0].item() / 1000.0
        # ref = median(block_rms) = 0 -> 1e-6, so ratio = 1e9 and the knee
        # argument saturates towards tanh(KNEE_STEEPNESS).
        ratio = 1000.0 / 1e-6
        expected = 1.0 - 0.5 * torch.tanh(torch.tensor(
            _harmonic.KNEE_STEEPNESS * (ratio - 24.0) / (ratio + _harmonic.RATIO_EPS))).item()
        self.assertAlmostEqual(gain, expected, places=4)

    def test_hard_ceiling_at_alpha_one(self):
        eq = _harmonic.harmonic_dynamic_eq
        x = torch.full((1, 16, 8), 1.0)
        x[0, 0, 0] = 1e6
        y = eq(x, 24.0, 1.0, 0.0, 0.0)
        # Steepness 3.0 pushes the saturation far enough that alpha = 1.0 lands
        # the extreme peak on the GAIN_MIN floor: a true -20 dB hard ceiling.
        self.assertAlmostEqual(y[0, 0, 0].item() / 1e6, _harmonic.GAIN_MIN, places=6)
        # Healthy lanes stay bit-exact while the outlier is clamped.
        self.assertTrue(torch.equal(y[0, 5, 5], x[0, 5, 5]))

    def test_low_boost_matches_curve(self):
        eq = _harmonic.harmonic_dynamic_eq
        x = torch.full((1, 8, 4), 1.0)
        x[0, 0, 0] = 1e-6
        y = eq(x, 24.0, 0.5, 0.2, 0.2)
        gain = y[0, 0, 0].item() / 1e-6
        self.assertAlmostEqual(gain, 1.0 + 0.2 * torch.tanh(torch.tensor((0.2 - 1e-6) / 0.2)).item(), places=4)
        # A healthy lane inside the same block must stay bit-exact.
        self.assertTrue(torch.equal(y[0, 3, 3], x[0, 3, 3]))

    def test_gain_safety_bounds(self):
        eq = _harmonic.harmonic_dynamic_eq
        x = torch.randn(2, 64, 32) * 3
        y = eq(x, 8.0, 1.0, 1.0, 1.0)
        m = x != 0
        g = y.float()[m] / x.float()[m]
        self.assertGreaterEqual(g.min().item(), 0.1 - 1e-6)
        self.assertLessEqual(g.max().item(), 2.0 + 1e-6)

    def test_degenerate_inputs_stay_safe(self):
        eq = _harmonic.harmonic_dynamic_eq
        z = torch.zeros(1, 10, 10)
        self.assertTrue(torch.equal(eq(z, 24.0, 0.5, 0.2, 0.2), z))
        self.assertTrue(torch.isfinite(eq(torch.full((1, 10, 10), 1e-30), 24.0, 0.5, 0.2, 0.2)).all())
        self.assertTrue(torch.isfinite(eq(torch.full((1, 10, 10), 1e6), 24.0, 0.5, 0.2, 0.2)).all())
        xn = torch.randn(1, 10, 10)
        xn[0, 0, 0] = float("nan")
        yn = eq(xn, 24.0, 0.5, 0.2, 0.2)
        self.assertTrue(torch.isnan(yn[0, 0, 0]))
        xi = torch.randint(0, 10, (1, 10, 10))
        self.assertTrue(torch.equal(xi, eq(xi, 24.0, 0.5, 0.2, 0.2)))
        self.assertEqual(eq([1, 2], 24.0, 0.5, 0.2, 0.2), [1, 2])


@unittest.skipUnless(torch is not None, "torch not available")
class TestHarmonicEQCfgPatch(unittest.TestCase):

    @staticmethod
    def _args(xin, cond_pred, uncond_pred, scale):
        args = {"cond": xin - cond_pred, "cond_scale": scale, "timestep": 1.0,
                "input": xin, "sigma": 1.0, "cond_denoised": cond_pred,
                "model": None, "model_options": {}, "input_cond": None, "input_uncond": None}
        if uncond_pred is None:
            args["uncond"] = None
            args["uncond_denoised"] = None
        else:
            args["uncond"] = xin - uncond_pred
            args["uncond_denoised"] = uncond_pred
        return args

    def test_transparent_matches_default_combine(self):
        patch = _harmonic.HarmonicEQCfgPatch(24.0, 0.0, 0.2, 0.0)
        torch.manual_seed(0)
        xin = torch.randn(2, 4, 32, 32)
        cond_pred = torch.randn(2, 4, 32, 32) * 0.8
        uncond_pred = torch.randn(2, 4, 32, 32) * 0.8
        default = uncond_pred + (cond_pred - uncond_pred) * 5.0
        # The CFG-function contract (cfg_result = input - f(args)) costs one
        # extra rounding step, so compare within a couple of ulps, not bitwise.
        cfg_result = xin - patch(self._args(xin, cond_pred, uncond_pred, 5.0))
        self.assertTrue(torch.allclose(cfg_result, default, rtol=0, atol=1e-5))

    def test_cond_only_fast_path(self):
        patch = _harmonic.HarmonicEQCfgPatch(24.0, 0.0, 0.2, 0.0)
        torch.manual_seed(1)
        xin = torch.randn(1, 4, 16, 16)
        cond_pred = torch.randn(1, 4, 16, 16)
        args = self._args(xin, cond_pred, None, 1.0)
        args["uncond_denoised"] = None
        self.assertTrue(torch.allclose(xin - patch(args), cond_pred, rtol=0, atol=1e-6))

    def test_fail_open_when_math_raises(self):
        patch = _harmonic.HarmonicEQCfgPatch(24.0, 0.5, 0.2, 0.2)
        torch.manual_seed(2)
        xin = torch.randn(1, 4, 16, 16)
        cond_pred = torch.randn(1, 4, 16, 16)
        uncond_pred = torch.randn(1, 4, 16, 16)
        default = uncond_pred + (cond_pred - uncond_pred) * 7.0
        original = _harmonic.harmonic_dynamic_eq

        def boom(*a, **k):
            raise RuntimeError("boom")

        _harmonic.harmonic_dynamic_eq = boom
        try:
            cfg_result = xin - patch(self._args(xin, cond_pred, uncond_pred, 7.0))
        finally:
            _harmonic.harmonic_dynamic_eq = original
        self.assertTrue(torch.allclose(cfg_result, default, rtol=0, atol=1e-5))

    def test_args_tensors_not_mutated(self):
        patch = _harmonic.HarmonicEQCfgPatch(24.0, 0.5, 0.2, 0.2)
        xin = torch.randn(1, 4, 8, 8)
        cond_pred = torch.randn(1, 4, 8, 8)
        uncond_pred = torch.randn(1, 4, 8, 8)
        args = self._args(xin, cond_pred, uncond_pred, 5.0)
        patch(args)
        self.assertTrue(torch.equal(args["input"], xin))
        self.assertTrue(torch.equal(args["cond_denoised"], cond_pred))
        self.assertTrue(torch.equal(args["uncond_denoised"], uncond_pred))


@unittest.skipUnless(torch is not None, "torch not available")
class TestHarmonicDynamicEqNode(unittest.TestCase):

    def test_widget_order_and_metadata(self):
        node = _harmonic.HarmonicDynamicEQ()
        req = node.INPUT_TYPES()["required"]
        self.assertEqual(list(req.keys()),
                         ["model", "threshold_high", "alpha_compress", "threshold_low", "beta_boost", "enabled"])
        self.assertEqual(req["threshold_high"][1]["default"], 24.0)
        self.assertEqual((req["threshold_high"][1]["min"], req["threshold_high"][1]["max"]), (8.0, 128.0))
        self.assertEqual(node.RETURN_TYPES, ("MODEL",))
        self.assertEqual(node.CATEGORY, "FeiFei")

    def test_patch_installs_on_clone_only(self):
        class FakePatcher:
            def __init__(self):
                self.model_options = {"transformer_options": {}}

            def clone(self):
                return FakePatcher()

            def set_model_sampler_cfg_function(self, f, disable_cfg1_optimization=False):
                self.model_options["sampler_cfg_function"] = f

        node = _harmonic.HarmonicDynamicEQ()
        base = FakePatcher()
        (patched,) = node.patch(base, 24.0, 0.5, 0.2, 0.2, True)
        self.assertIsNot(patched, base)
        self.assertIsInstance(patched.model_options.get("sampler_cfg_function"),
                              _harmonic.HarmonicEQCfgPatch)
        self.assertIsNone(base.model_options.get("sampler_cfg_function"))
        (plain,) = node.patch(base, 24.0, 0.5, 0.2, 0.2, False)
        self.assertNotIn("sampler_cfg_function", plain.model_options)

    def test_mappings_registered(self):
        self.assertIn("HarmonicDynamicEQ", _harmonic.NODE_CLASS_MAPPINGS)
        self.assertIn("HarmonicDynamicEQ", _harmonic.NODE_DISPLAY_NAME_MAPPINGS)


@unittest.skipUnless(torch is not None and torch.cuda.is_available(), "CUDA not available")
class TestHarmonicDynamicEqCuda(unittest.TestCase):

    def test_cuda_bf16_and_bit_exact_band(self):
        eq = _harmonic.harmonic_dynamic_eq
        x = (torch.randn(2, 128, 640, device="cuda") * 0.7).to(torch.bfloat16)
        y = eq(x, 24.0, 0.5, 0.2, 0.2)
        self.assertEqual(y.dtype, torch.bfloat16)
        self.assertTrue(torch.isfinite(y.float()).all())
        band = torch.full((2, 100, 32), 0.5, device="cuda", dtype=torch.bfloat16)
        self.assertTrue(torch.equal(band, eq(band, 24.0, 0.5, 0.2, 0.2)))


if __name__ == "__main__":
    unittest.main()
