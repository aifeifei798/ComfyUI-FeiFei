"""HarmonicDynamicEQ - audio-mastering style dynamic range shaping for diffusion models.

A MODEL patch node: wire it between the model loader (or any MODEL chain) and the
sampler. It never touches checkpoint weights: the effect lives only on the cloned
ModelPatcher this node returns, so bypassing the node (or wiring the original
MODEL) restores the model bit-for-bit.

The transfer curve mimics a mastering compander applied to the CFG-combined
denoised prediction (the latent the sampler consumes on every step):

* High knee (down-compression / limiter): activation magnitudes above
  ``threshold_high`` x the robust reference level are smoothly pulled back with a
  tanh soft knee - the self-regulating cure for high-CFG burn-out (blown
  highlights, crushed blacks, waxy over-exposure), because CFG amplification is
  exactly what pushes those outliers past the knee.
* Low knee (up-expansion / AGC boost): magnitudes below ``threshold_low`` x the
  reference get a gentle gain lift, recovering dark detail and contrast on flat,
  murky renders.
* Healthy band: gain is exactly 1.0, so the model output passes through
  bit-identically - no drift, no color shift, no added noise.

Hook choice: this installs through ``model.set_model_sampler_cfg_function``. The
attention-output patch (``set_model_attn2_output_patch`` / ``set_model_attn2_patch``)
only fires inside SD-style ``BasicTransformerBlock`` (comfy/ldm/modules/attention.py);
Flux and other DiT models route through comfy/ldm/flux/layers.py, which applies
``attn1_output_patch`` only - an attn2 patch would silently do nothing there.
The CFG-function hook is architecture-agnostic (SD1.5 / SDXL / Flux / any DiT),
runs once per sampling step on a small latent tensor, and replicates ComfyUI's
default combination exactly before shaping it:

    denoised = uncond + (cond - uncond) * cond_scale   # ComfyUI default
    return input - eq(denoised)                        # contract: cfg_result = input - f(args)

Because the CFG-function slot is a single slot (same as the built-in RescaleCFG
node), chaining two HarmonicDynamicEQ instances makes the later one replace the
earlier one's CFG function.

All math is pure PyTorch vectorized broadcast computation - no Python per-element
loops, no learned state, no growing buffers; the output keeps the input dtype
(bf16 / fp16 / fp32). Any failure inside the shaping math fails open to the
untouched default CFG result instead of raising into the workflow.
"""

import torch

# Slope of the tanh knees. Fixed by design (not a widget): as ratio -> inf the
# high knee saturates towards tanh(KNEE_STEEPNESS), capping the maximum cut at
# alpha_compress * tanh(KNEE_STEEPNESS) ~= 0.76 * alpha_compress.
KNEE_STEEPNESS = 1.0

# Hard safety clip for the combined gain (linear units); the knee formulas alone
# stay within (0.24, 1.77), so this only guards against pathological inputs.
GAIN_MIN = 0.1
GAIN_MAX = 2.0

# Keeps the division alive when every block is perfectly silent.
REF_EPS = 1e-6


def harmonic_dynamic_eq(x, threshold_high, alpha_compress, threshold_low, beta_boost):
    """Apply the compander curve to ``x`` (any (B, ...) activation tensor).

    The reference level is the per-sample median of per-block RMS, where a block
    is one token for (B, S, D) tensors, one spatial position for (B, C, H, W)
    latents (and one temporal-spatial position for (B, C, T, H, W) video). The
    median keeps a few hot blocks from inflating the reference, which would mute
    the limiter exactly where it is needed. Returns a tensor with the same shape
    and dtype; lanes inside the healthy band are returned bit-exact because their
    gain is exactly 1.0.
    """
    if not isinstance(x, torch.Tensor) or x.ndim < 2 or not x.is_floating_point():
        return x
    orig_dtype = x.dtype

    # fp32 working copy for statistics and gains; never mutated, so the fp32
    # fast path of Tensor.to() (which may alias x) stays safe.
    x32 = x.to(torch.float32)

    # Per-block RMS: features live on dim 1 for latent-style tensors and on the
    # last dim for attention/token-style tensors.
    if x.ndim >= 4:
        block_rms = x32.pow(2).mean(dim=1).sqrt()
    else:
        block_rms = x32.pow(2).mean(dim=-1).sqrt()
    if block_rms.ndim > 1:
        block_rms = block_rms.flatten(1)
        ref = block_rms.median(dim=1).values + REF_EPS
    else:
        ref = block_rms + REF_EPS
    ref = ref.reshape(-1, *([1] * (x.ndim - 1)))

    # Outlier ratio against the robust reference, elementwise.
    ratio = x32.abs() / ref

    gain = torch.ones_like(ratio)
    if alpha_compress > 0.0:
        # (ratio - th) / ratio -> 0 at the knee, -> 1 as ratio -> inf, so the
        # where() boundary is continuous and the cut saturates smoothly.
        over = torch.tanh(KNEE_STEEPNESS * (ratio - threshold_high) / ratio)
        gain = torch.where(ratio > threshold_high, 1.0 - alpha_compress * over, gain)
    if threshold_low > 0.0 and beta_boost > 0.0:
        under = torch.tanh((threshold_low - ratio) / threshold_low)
        gain = torch.where(ratio < threshold_low, 1.0 + beta_boost * under, gain)

    gain = gain.clamp(min=GAIN_MIN, max=GAIN_MAX).to(orig_dtype)
    return x * gain


class HarmonicEQCfgPatch:
    """Callable installed as the sampler CFG function.

    Contract (comfy/samplers.py ``cfg_function``): the return value R is consumed
    as ``cfg_result = input - R``, so R must be the noisy input minus the final
    denoised prediction. The default combination is replicated first, then the
    harmonic EQ shapes it. Everything is out-of-place: the caller's tensors are
    never mutated.
    """

    def __init__(self, threshold_high, alpha_compress, threshold_low, beta_boost):
        self.threshold_high = float(threshold_high)
        self.alpha_compress = float(alpha_compress)
        self.threshold_low = float(threshold_low)
        self.beta_boost = float(beta_boost)

    def __call__(self, args):
        try:
            return self._shaped(args)
        except Exception:
            # Fail open to ComfyUI's default CFG combination; only a changed
            # ComfyUI contract (missing keys) may bubble up.
            x = args["input"]
            cond = args["cond_denoised"]
            uncond = args.get("uncond_denoised")
            if isinstance(uncond, torch.Tensor):
                return x - (uncond + (cond - uncond) * args.get("cond_scale", 1.0))
            return x - cond

    def _shaped(self, args):
        x = args["input"]
        cond_denoised = args["cond_denoised"]
        uncond_denoised = args.get("uncond_denoised")
        cond_scale = args.get("cond_scale", 1.0)
        if isinstance(uncond_denoised, torch.Tensor):
            denoised = uncond_denoised + (cond_denoised - uncond_denoised) * cond_scale
        else:
            # CFG = 1.0 fast path: no uncond prediction exists.
            denoised = cond_denoised
        try:
            denoised = harmonic_dynamic_eq(
                denoised, self.threshold_high, self.alpha_compress, self.threshold_low, self.beta_boost)
        except Exception:
            pass  # fail open: fall back to the untouched CFG result
        return x - denoised


class HarmonicDynamicEQ:
    """MODEL patch node: harmonic dynamic EQ / compander on the sampling trajectory."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "threshold_high": ("FLOAT", {
                    "default": 24.0, "min": 8.0, "max": 128.0, "step": 0.5,
                    "tooltip": "Limiter knee, in multiples of the robust reference level. Magnitudes above threshold_high * ref are smoothly pulled back; lower it (8-16) for stronger protection against high-CFG burn-out."}),
                "alpha_compress": ("FLOAT", {
                    "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Down-compression depth of the high knee. 0 disables the limiter; 1 applies the strongest peak control."}),
                "threshold_low": ("FLOAT", {
                    "default": 0.2, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Boost knee, in multiples of the reference level. Magnitudes below threshold_low * ref receive a gentle gain lift; raise it to lift more of the dark range."}),
                "beta_boost": ("FLOAT", {
                    "default": 0.2, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Up-expansion depth of the low knee. 0 disables the boost; 1 applies the strongest low-level lift for dark, flat renders."}),
                "enabled": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Master switch. When off, an untouched model clone is returned (equivalent to bypassing the node)."}),
            },
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "patch"
    CATEGORY = "FeiFei"
    DESCRIPTION = ("Audio-mastering style compander (limiter + AGC boost) on the CFG-combined "
                   "prediction. Bit-exact pass-through in the healthy band; works on SD1.5/SDXL, "
                   "Flux and any DiT. High-CFG burn and dark flat renders respond to the two knees.")

    def patch(self, model, threshold_high, alpha_compress, threshold_low, beta_boost, enabled):
        patched = model.clone()
        if enabled:
            patched.set_model_sampler_cfg_function(
                HarmonicEQCfgPatch(threshold_high, alpha_compress, threshold_low, beta_boost))
        return (patched,)


NODE_CLASS_MAPPINGS = {
    "HarmonicDynamicEQ": HarmonicDynamicEQ,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "HarmonicDynamicEQ": "FeiFei Harmonic Dynamic EQ",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "HarmonicDynamicEQ", "harmonic_dynamic_eq"]
