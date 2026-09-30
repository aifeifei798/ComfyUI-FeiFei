"""Physical film and "breathing room" post-processing: lens -> film base ->
development -> emulsion, a four-stage pipeline.

Pure torch, no external models or heavy libraries. The stage order follows real
film processing: lens (vignette / lateral chromatic aberration) -> film-base
halation -> development S-curve and split toning -> micro contrast -> emulsion
grain. Grain goes last so the development curve cannot crush it away.
"""

import math

import torch
import torch.nn.functional as F

# Every size-dependent trait is referenced to a 1024 short side, so grain,
# halation and micro contrast scale automatically with the frame
RES_REF = 1024.0

# Grain strength conversion: the default grain_amount=0.25 lands near +-3.5/255
# (the magnitude of a real scan), and 1.0 reaches about +-14/255. Past that it
# stops reading as film and starts reading as TV snow.
GRAIN_GAIN = 0.10
# Share of per-pixel detail inside a grain clump. Too much degenerates into
# salt-and-pepper noise, so it stays low to keep clumps looking like silver halide
GRAIN_FINE_MIX = 0.06
# At a 1024 short side, grain_size=1.0 is a ~2.5px clump, close to a real film scan
GRAIN_CELL_SCALE = 2.5

LUMA_WEIGHTS = (0.2126, 0.7152, 0.0722)
HALATION_TINT = (1.0, 0.62, 0.42)

# Baseline character shared by most stocks; each film only overrides what differs
_BASE_TRAITS = {
    "grain_shadows": 0.55,
    "grain_chroma": 0.20,
    "halation": 0.10,
    "halation_threshold": 0.80,
    "halation_radius": 1.0,
    "halation_tint": HALATION_TINT,
    "vignette": 0.10,
    "vignette_size": 0.75,
    "micro_contrast": 0.15,
    "chroma_shift": 0.8,
    "split_tone": 0.05,
}

# First dropdown: film categories. Digital Clean / expired stock are not "a
# particular roll of film", so they get their own Other bucket instead of being
# mixed in with real stocks. Every value visible in the UI is English.
FILM_TYPES = [
    "Color Negative",
    "Slide",
    "B&W Negative",
    "Cinema",
    "Special",
    "Other",
]

# Preset and manual are strictly either/or with no middle state: in Preset mode
# no slider is read, in Custom mode neither dropdown is read. Deliberately no
# "preset with slider tweaks" hybrid, so it is never ambiguous which one is
# actually driving the result.
FILM_MODES = ["Preset", "Custom"]

# The only knob in Preset mode scales "effect amount" and leaves the traits that
# define the film's character (grain size, shadow bias, halation threshold, ...)
# alone, so gain=0.5 is this film at half strength, not this film at half
# parameters. These 7 are exactly the ones film_finish gates on > 0, so scaling
# them all to 0 returns the untouched input.
_GAINABLE = ("grain_amount", "halation", "vignette", "tone", "micro_contrast", "split_tone", "chroma_shift")

# The second dropdown is grouped by category, and the grouping order is the
# dropdown order. Each film lists only what differs from _BASE_TRAITS.
FILM_PRESETS = {
    # ── Color Negative ── everyday portraits and street work
    "Portra 160": {"grain_amount": 0.12, "grain_size": 0.6, "grain_shadows": 0.50, "tone": 0.14, "micro_contrast": 0.08},
    "Portra 400": {"grain_amount": 0.16, "grain_size": 0.7, "grain_shadows": 0.50, "tone": 0.18, "micro_contrast": 0.10, "halation_threshold": 0.82, "halation_radius": 0.8, "vignette_size": 0.75, "chroma_shift": 0.8},
    "Portra 800": {"grain_amount": 0.24, "grain_size": 0.95, "grain_shadows": 0.55, "tone": 0.19, "micro_contrast": 0.11, "halation_threshold": 0.82, "halation_radius": 0.8, "vignette_size": 0.75},
    "Ektar 100": {"grain_amount": 0.14, "grain_size": 0.6, "grain_shadows": 0.45, "tone": 0.30, "micro_contrast": 0.22, "split_tone": 0.03},
    "Ektar 500": {"grain_amount": 0.28, "grain_size": 1.0, "grain_shadows": 0.50, "tone": 0.32, "micro_contrast": 0.22, "split_tone": 0.03},
    "Gold 200": {"grain_amount": 0.24, "grain_size": 1.1, "grain_shadows": 0.55, "tone": 0.28, "micro_contrast": 0.18, "split_tone": 0.07},
    "Gold 400": {"grain_amount": 0.30, "grain_size": 1.3, "grain_shadows": 0.58, "tone": 0.30, "micro_contrast": 0.19, "split_tone": 0.08},
    "Fuji C200": {"grain_amount": 0.22, "grain_size": 1.0, "grain_shadows": 0.52, "tone": 0.34, "micro_contrast": 0.20, "split_tone": 0.06},
    "Pro 400H": {"grain_amount": 0.24, "grain_size": 1.0, "grain_shadows": 0.62, "tone": 0.30, "micro_contrast": 0.20, "halation_tint": (0.90, 0.96, 1.00), "split_tone": 0.10},

    # ── Slide ── reversal film: extremely fine grain, very high contrast
    "Ektachrome E100": {"grain_amount": 0.05, "grain_size": 0.4, "grain_shadows": 0.40, "grain_chroma": 0.10, "tone": 0.46, "micro_contrast": 0.28, "halation": 0.05, "halation_threshold": 0.86, "halation_radius": 0.8, "vignette": 0.12, "split_tone": 0.03},
    "Fuji Provia 100F": {"grain_amount": 0.07, "grain_size": 0.5, "grain_shadows": 0.42, "grain_chroma": 0.12, "tone": 0.42, "micro_contrast": 0.26, "halation": 0.06, "split_tone": 0.03},

    # ── B&W Negative ── grain is the star, colour grain pushed to the minimum
    "Tri-X 400": {"grain_amount": 0.38, "grain_size": 1.8, "grain_shadows": 0.72, "grain_chroma": 0.12, "tone": 0.35, "micro_contrast": 0.22, "halation": 0.05, "halation_threshold": 0.86, "halation_radius": 0.7, "vignette": 0.06, "vignette_size": 0.80, "chroma_shift": 1.2, "split_tone": 0.03},
    "HP5 Plus 400": {"grain_amount": 0.34, "grain_size": 1.4, "grain_shadows": 0.68, "grain_chroma": 0.10, "tone": 0.30, "micro_contrast": 0.20, "halation": 0.05, "vignette": 0.07, "chroma_shift": 1.0, "split_tone": 0.03},
    "FP4 Plus 125": {"grain_amount": 0.30, "grain_size": 1.25, "grain_shadows": 0.64, "grain_chroma": 0.10, "tone": 0.28, "micro_contrast": 0.18, "halation": 0.05, "split_tone": 0.04},
    "TMax 400": {"grain_amount": 0.20, "grain_size": 0.6, "grain_shadows": 0.55, "grain_chroma": 0.08, "tone": 0.32, "micro_contrast": 0.24, "halation": 0.04},
    "TMax 100": {"grain_amount": 0.16, "grain_size": 0.5, "grain_shadows": 0.52, "grain_chroma": 0.08, "tone": 0.28, "micro_contrast": 0.22, "halation": 0.04},
    "Delta 3200": {"grain_amount": 0.18, "grain_size": 0.55, "grain_shadows": 0.58, "grain_chroma": 0.10, "tone": 0.30, "micro_contrast": 0.22, "halation": 0.04},
    "Delta 100": {"grain_amount": 0.14, "grain_size": 0.45, "grain_shadows": 0.50, "grain_chroma": 0.08, "tone": 0.26, "micro_contrast": 0.20, "halation": 0.04},
    "Acros 100": {"grain_amount": 0.15, "grain_size": 0.5, "grain_shadows": 0.50, "grain_chroma": 0.08, "tone": 0.28, "micro_contrast": 0.20, "halation": 0.04, "split_tone": 0.04},

    # ── Cinema ── CineStill's red halation is the star
    "CineStill 800T": {"grain_amount": 0.22, "grain_size": 1.2, "grain_shadows": 0.60, "grain_chroma": 0.30, "tone": 0.28, "micro_contrast": 0.12, "halation": 0.34, "halation_threshold": 0.70, "halation_radius": 1.8, "halation_tint": (1.0, 0.42, 0.34), "vignette": 0.16, "vignette_size": 0.70, "chroma_shift": 1.0, "split_tone": 0.09},
    "CineStill 400D": {"grain_amount": 0.20, "grain_size": 1.0, "grain_shadows": 0.58, "tone": 0.26, "micro_contrast": 0.14, "halation": 0.26, "halation_threshold": 0.72, "halation_radius": 1.5, "vignette": 0.14, "vignette_size": 0.70, "split_tone": 0.08},
    "Vision3 250D": {"grain_amount": 0.18, "grain_size": 0.8, "grain_shadows": 0.50, "tone": 0.24, "micro_contrast": 0.16, "halation": 0.08},
    "Vision3 500T": {"grain_amount": 0.28, "grain_size": 1.2, "grain_shadows": 0.58, "tone": 0.28, "micro_contrast": 0.18, "halation": 0.12, "vignette": 0.12},
    "Cine 50D": {"grain_amount": 0.16, "grain_size": 0.7, "grain_shadows": 0.48, "tone": 0.22, "micro_contrast": 0.14, "halation": 0.08},
    "500T Expired": {"grain_amount": 0.40, "grain_size": 1.9, "grain_shadows": 0.68, "tone": 0.34, "micro_contrast": 0.16, "halation": 0.22, "halation_threshold": 0.74, "halation_radius": 1.5, "halation_tint": (1.0, 0.56, 0.46), "vignette": 0.14, "split_tone": 0.10},

    # ── Special ── strong-tasting looks, not standard stocks
    "Cinestack 800T": {"grain_amount": 0.20, "grain_size": 1.1, "halation": 0.60, "halation_threshold": 0.55, "halation_radius": 2.6, "vignette": 0.20, "split_tone": 0.12},
    "Push +2 Stops": {"grain_amount": 0.52, "grain_size": 2.2, "grain_shadows": 0.72, "tone": 0.42, "micro_contrast": 0.24, "halation": 0.14, "vignette": 0.14},
    "Cross Process": {"grain_amount": 0.26, "grain_size": 1.0, "tone": 0.40, "micro_contrast": 0.26, "halation": 0.18, "halation_tint": (1.0, 0.52, 0.60), "split_tone": 0.12},

    # ── Other ── reference group and the digital look, handy for A/B comparison
    "Digital Clean": {"grain_amount": 0.05, "grain_size": 0.6, "grain_shadows": 0.40, "grain_chroma": 0.15, "tone": 0.10, "micro_contrast": 0.06, "halation": 0.05, "halation_threshold": 0.85, "halation_radius": 0.8, "vignette": 0.04, "vignette_size": 0.80, "chroma_shift": 0.4, "split_tone": 0.02},
}

# Which category each film belongs to in the first dropdown
_PRESET_TYPE = {
    "Portra 160": "Color Negative", "Portra 400": "Color Negative", "Portra 800": "Color Negative",
    "Ektar 100": "Color Negative", "Ektar 500": "Color Negative", "Gold 200": "Color Negative",
    "Gold 400": "Color Negative", "Fuji C200": "Color Negative", "Pro 400H": "Color Negative",
    "Ektachrome E100": "Slide", "Fuji Provia 100F": "Slide",
    "Tri-X 400": "B&W Negative", "HP5 Plus 400": "B&W Negative", "FP4 Plus 125": "B&W Negative",
    "TMax 400": "B&W Negative", "TMax 100": "B&W Negative", "Delta 3200": "B&W Negative",
    "Delta 100": "B&W Negative", "Acros 100": "B&W Negative",
    "CineStill 800T": "Cinema", "CineStill 400D": "Cinema", "Vision3 250D": "Cinema",
    "Vision3 500T": "Cinema", "Cine 50D": "Cinema", "500T Expired": "Cinema",
    "Cinestack 800T": "Special", "Push +2 Stops": "Special", "Cross Process": "Special",
    "Digital Clean": "Other",
}

# Options for the second dropdown: the whole table reordered by FILM_TYPES
PRESET_ORDER = [
    name for film_type in FILM_TYPES for name, kind in _PRESET_TYPE.items() if kind == film_type
]


def _resolve_params(preset, gain, sliders):
    """The preset takes over wholesale, or the sliders do. Pure function for unit tests.

    A None preset (Custom mode) returns the slider values untouched; otherwise
    every value comes from the preset and no slider value leaks in, so the two
    modes never blend. gain only applies on the preset path.
    """
    if preset is None:
        return dict(sliders, halation_tint=HALATION_TINT)
    params = {**sliders, **_BASE_TRAITS, **FILM_PRESETS[preset]}
    if gain < 1.0:
        for key in _GAINABLE:
            params[key] *= gain
    return params


def _luma(img):
    """Rec.709 luma; zero-pads and renormalizes when there are fewer channels"""
    channels = img.shape[-1]
    weights = list(LUMA_WEIGHTS[:channels])
    weights += [0.0] * (channels - len(weights))
    w = torch.tensor(weights, device=img.device, dtype=img.dtype)
    return (img * (w / w.sum())).sum(-1, keepdim=True)


def _radial(height, width, device, dtype):
    """Normalized radius: 1 at the half-width/half-height, ~1.41 at the corners"""
    y = torch.linspace(-1.0, 1.0, height, device=device, dtype=dtype).view(1, height, 1)
    x = torch.linspace(-1.0, 1.0, width, device=device, dtype=dtype).view(1, 1, width)
    return (x * x + y * y).sqrt()


def _blur(plane, sigma):
    """Separable Gaussian with reflect padding; plane: [B,C,H,W]"""
    if sigma <= 0.5:
        return plane
    half = int(sigma * 3)
    steps = torch.arange(-half, half + 1, device=plane.device, dtype=plane.dtype)
    kernel = torch.exp(-0.5 * (steps / sigma) ** 2)
    kernel = kernel / kernel.sum()
    channels = plane.shape[1]
    plane = F.pad(plane, (half, half, 0, 0), mode="reflect")
    plane = F.conv2d(plane, kernel.view(1, 1, 1, -1).expand(channels, 1, 1, -1), groups=channels)
    plane = F.pad(plane, (0, 0, half, half), mode="reflect")
    return F.conv2d(plane, kernel.view(1, 1, -1, 1).expand(channels, 1, -1, 1), groups=channels)


def _vignette(img, radial, amount, size):
    """size is the fraction of the radius where falloff starts; corners bottom out
    at 1-amount and never go fully black"""
    falloff = ((radial / size - 1.0) / 0.4142).clamp(0, 1) ** 1.8
    return img * (1.0 - amount * falloff).unsqueeze(-1)


def _chroma_shift(img, radial, amount_px):
    """Red and blue scale outward with radius while green stays put: lateral
    chromatic aberration, nearly invisible at the centre and edge midpoints"""
    height, width = img.shape[1:3]
    edge = (radial / 1.4142) ** 2
    y = torch.linspace(-1.0, 1.0, height, device=img.device, dtype=img.dtype).view(height, 1).expand(height, width)
    x = torch.linspace(-1.0, 1.0, width, device=img.device, dtype=img.dtype).view(1, width).expand(height, width)
    zoom = 1.0 + (amount_px * 2.0 / RES_REF) * edge
    grid = torch.stack([x / zoom, y / zoom], dim=-1)

    out = []
    for index, channel in enumerate(img.unbind(dim=-1)):
        if index == 1:
            out.append(channel)
            continue
        warped = F.grid_sample(
            channel.unsqueeze(1), grid.expand(channel.shape[0], -1, -1, -1),
            mode="bilinear", padding_mode="border", align_corners=True,
        )
        out.append(warped.squeeze(1))
    return torch.stack(out, dim=-1)


def _halation(img, amount, threshold, radius, tint):
    """Warm glow from highlights scattering through the film base: blurred at
    reduced resolution, then screen-blended"""
    height, width = img.shape[1:3]
    mask = ((_luma(img) - threshold) / max(1e-3, 1.0 - threshold)).clamp(0, 1) ** 1.5
    sigma = radius * min(height, width) / RES_REF * 8.0
    # Halation is a low-frequency effect: blur at a working resolution around
    # sigma=6, so a large radius on 4K does not get slow
    down = min(16, max(2, int(round(sigma / 6.0))))
    work = mask.permute(0, 3, 1, 2)
    small = F.interpolate(work, size=(max(1, height // down), max(1, width // down)), mode="area")
    glow = _blur(small, sigma / down)
    glow = F.interpolate(glow, size=(height, width), mode="bilinear", align_corners=False)
    tint_t = torch.tensor(list(tint)[:img.shape[-1]], device=img.device, dtype=img.dtype)
    return 1.0 - (1.0 - img) * (1.0 - glow.permute(0, 2, 3, 1) * tint_t * amount)


def _tone(img, amount, curve=0.35, lift=0.015, mid=0.18):
    """Development S-curve: expands midtones around middle grey and compresses
    shadows and highlights in log-density space.

    The sine term rolls the slope back off at +-L and conserves both ends exactly,
    so pure white stays 1.0 and middle grey is untouched. Highlights are therefore
    only shoulder-compressed, never pushed out of frame, which a polynomial
    S-curve cannot guarantee.
    """
    stops = -math.log(mid)
    ld = (torch.log(_luma(img).clamp(1e-4, 1.0)) - math.log(mid)).clamp(-stops, stops)
    shaped = ld + curve * (stops / math.pi) * torch.sin(math.pi * ld / stops)
    graded = img * torch.exp(shaped - ld) + lift * (1.0 - img)
    return img + (graded - img) * amount


def _split_tone(img, amount):
    """Warm shadows, cool highlights, in small doses, just to take the digital edge off"""
    luma = _luma(img).clamp(0.0, 1.0)
    channels = img.shape[-1]
    warm = torch.tensor([1.06, 1.0, 0.94][:channels], device=img.device, dtype=img.dtype)
    cool = torch.tensor([0.97, 1.0, 1.05][:channels], device=img.device, dtype=img.dtype)
    return img * (1.0 + ((warm * (1.0 - luma) + cool * luma) - 1.0) * amount)


def _micro_contrast(img, amount):
    """Small-radius unsharp on luma only, so it adds no colour fringes; puts the
    skin and fabric back after the waxy look"""
    luma = _luma(img)
    sigma = 1.2 * min(img.shape[1], img.shape[2]) / RES_REF
    base = _blur(luma.permute(0, 3, 1, 2), sigma).permute(0, 2, 3, 1)
    return img + (luma - base) * amount


def _grain_field(height, width, channels, cell_px, seed, device, dtype):
    """Clumped grain from upsampling a low-resolution noise field, mixed with a
    little per-pixel detail, then normalized"""
    grid_h = max(16, int(round(height / cell_px)))
    grid_w = max(16, int(round(width / cell_px)))
    generator = torch.Generator(device=device).manual_seed(int(seed))
    field = torch.randn(1, channels, grid_h, grid_w, generator=generator, device=device, dtype=torch.float32)
    field = F.interpolate(field, (height, width), mode="bicubic", align_corners=False)
    if GRAIN_FINE_MIX > 0:
        detail = torch.randn(1, channels, height, width, generator=generator, device=device, dtype=torch.float32)
        field = field * (1.0 - GRAIN_FINE_MIX) + detail * GRAIN_FINE_MIX
    return (field / field.std()).to(dtype).permute(0, 2, 3, 1)


def _grain(img, amount, size, shadows, chroma, seed):
    """Grain strength weighted by luminance: strongest in the midtones, leaning
    into the shadows per `shadows`, decaying fast in the highlights"""
    luma = _luma(img)
    up = (luma / 0.5).clamp(0, 1) ** (0.35 + (1.0 - shadows) * 1.3)
    down = ((1.0 - luma) / 0.5).clamp(0, 1) ** 1.6
    response = torch.where(luma < 0.5, up, down)

    height, width, channels = img.shape[1], img.shape[2], img.shape[3]
    cell = max(1.0, size * GRAIN_CELL_SCALE * min(height, width) / RES_REF)
    field = _grain_field(height, width, channels, cell, seed, img.device, img.dtype)
    if chroma < 1.0:
        # chroma=0 is fully monochrome, =1 gives independent channels; the analytic
        # compensation keeps total amplitude independent of chroma
        mono = field.mean(dim=-1, keepdim=True)
        field = (mono + (field - mono) * chroma) * (3.0 / (1.0 + 2.0 * chroma ** 2)) ** 0.5
    return img + field * response * (amount * GRAIN_GAIN)


def film_finish(frame, params, seed):
    """Process a single frame; frame: [H,W,C] float tensor. Pure function for unit tests"""
    img = frame.unsqueeze(0)
    if params["vignette"] > 0 or params["chroma_shift"] > 0:
        radial = _radial(img.shape[1], img.shape[2], img.device, img.dtype)
        if params["vignette"] > 0:
            img = _vignette(img, radial, params["vignette"], params["vignette_size"])
        if params["chroma_shift"] > 0:
            img = _chroma_shift(img, radial, params["chroma_shift"])
    if params["halation"] > 0:
        img = _halation(img, params["halation"], params["halation_threshold"], params["halation_radius"], params["halation_tint"])
    if params["tone"] > 0:
        img = _tone(img, params["tone"])
    if params["split_tone"] > 0:
        img = _split_tone(img, params["split_tone"])
    if params["micro_contrast"] > 0:
        img = _micro_contrast(img, params["micro_contrast"])
    if params["grain_amount"] > 0:
        img = _grain(img, params["grain_amount"], params["grain_size"], params["grain_shadows"], params["grain_chroma"], seed)
    return img.squeeze(0)


class FeiFeiFilmGrainTone:
    """
    Physical film and "breathing room" post-processing: organic film grain +
    faint halation + dark-corner dispersion + development S-curve + micro
    contrast, to take the "too clean, too plasticky" look off AI images.
    """

    @classmethod
    def INPUT_TYPES(cls):
        # Panel order follows the order you use them in: pick the mode first, in
        # Preset mode the three dropdowns set everything along with the strength,
        # then the five main sliders, and seed last. Fine-tuning stays optional
        # and is grouped by film stage so it lines up with the main sliders.
        return {
            "required": {
                "image": ("IMAGE",),
                "mode": (FILM_MODES, {
                    "default": "Preset",
                    "tooltip": "Preset or custom, never a blend. "
                               "Preset = the film stock below drives every effect, all sliders are ignored "
                               "and preset_gain sets how strong it is. "
                               "Custom = the sliders below drive everything and film_type / preset / preset_gain "
                               "are ignored. Sliders always stay visible in Preset mode, they just have no effect there.",
                }),
                "film_type": (FILM_TYPES, {
                    "default": "Color Negative",
                    "tooltip": "Film family. Only groups the preset list below - it applies no effect of its own, "
                               "and is ignored entirely in Custom mode.",
                }),
                "preset": (PRESET_ORDER, {
                    "default": "Portra 400",
                    "tooltip": "Film stock to imitate. Only grain size, grain response, halation and lens "
                               "falloff are changed - never a colour grade, so the image keeps its own colours. "
                               "Ignored in Custom mode, where the sliders take over instead.",
                }),
                "preset_gain": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "How strong the preset is, Preset mode only. Scales grain, halation, vignette, "
                               "print curve, micro-contrast, split tone and colour fringing together, while the "
                               "film's own character - grain size, shadow bias, glow threshold - stays put. "
                               "0.0 leaves the image untouched, 0.5 is a half-strength version of the same stock, "
                               "not half of its parameters. Ignored in Custom mode.",
                }),
                "grain_amount": ("FLOAT", {
                    "default": 0.25, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Grain strength. Default is about +/-3.5/255, the level of a real film scan. "
                               "Above roughly 0.6 it starts to look like TV snow instead of film.",
                }),
                "grain_size": ("FLOAT", {
                    "default": 1.0, "min": 0.2, "max": 4.0, "step": 0.05,
                    "tooltip": "Size of the grain clumps, scaled automatically with the frame size. "
                               "1.0 is about 2.5px clumps on a 1024px image; 4.0 is coarse push-processed grain.",
                }),
                "halation": ("FLOAT", {
                    "default": 0.15, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Warm glow bleeding out of the highlights. Keep it subtle (0.1-0.2) for a normal "
                               "film look; push to 0.3-0.6 for a light-leak or CineStill feel.",
                }),
                "vignette": ("FLOAT", {
                    "default": 0.12, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Corner falloff. The corners darken by at most this amount, so 0.12 is a gentle edge.",
                }),
                "tone": ("FLOAT", {
                    "default": 0.25, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Print S-curve: expands the midtones and holds the highlights. "
                               "Includes a small black lift so shadows never reach pure black.",
                }),
                "seed": ("INT", {
                    "default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True,
                    "tooltip": "Same seed gives the same grain. Each image in a batch gets its own grain, "
                               "so animating a frame sequence will not flicker.",
                }),
            },
            "optional": {
                "grain_shadows": ("FLOAT", {
                    "default": 0.55, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Where the grain lives. Low keeps it in the midtones, high pushes it into the "
                               "shadows for a pushed-film or high-ISO look.",
                }),
                "grain_chroma": ("FLOAT", {
                    "default": 0.25, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "0 = fully monochrome grain, 1 = each channel gets its own grain. "
                               "Real film sits low here; black and white film is near 0.08.",
                }),
                "halation_threshold": ("FLOAT", {
                    "default": 0.78, "min": 0.4, "max": 1.0, "step": 0.01,
                    "tooltip": "How bright a pixel must be before it starts to bleed. "
                               "Lower it to make the glow spread from mid-bright areas too.",
                }),
                "halation_radius": ("FLOAT", {
                    "default": 1.0, "min": 0.2, "max": 4.0, "step": 0.05,
                    "tooltip": "Spread of the glow, scaled with the frame size. "
                               "CineStill-style halation wants 1.5-2.0.",
                }),
                "vignette_size": ("FLOAT", {
                    "default": 0.72, "min": 0.3, "max": 1.4, "step": 0.01,
                    "tooltip": "How far in the darkening starts. Small values darken more of the frame.",
                }),
                "chroma_shift": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 8.0, "step": 0.1,
                    "tooltip": "Lateral chromatic aberration, in pixels of fringing at the corner "
                               "of a 1024px image. Invisible in the centre.",
                }),
                "micro_contrast": ("FLOAT", {
                    "default": 0.15, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Fine detail sharpening on luminance only, no colour fringes. "
                               "The single most effective slider against the waxy AI-skin look.",
                }),
                "split_tone": ("FLOAT", {
                    "default": 0.06, "min": 0.0, "max": 0.3, "step": 0.01,
                    "tooltip": "Warm shadows and cool highlights. A very small amount, "
                               "just enough to stop the colours looking purely digital.",
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "apply_film"
    CATEGORY = "FeiFei"

    def apply_film(
        self,
        image,
        mode,
        film_type,
        preset,
        preset_gain,
        grain_amount,
        grain_size,
        halation,
        vignette,
        tone,
        seed,
        grain_shadows=0.55,
        grain_chroma=0.25,
        halation_threshold=0.78,
        halation_radius=1.0,
        vignette_size=0.72,
        chroma_shift=1.0,
        micro_contrast=0.15,
        split_tone=0.06,
    ):
        if not isinstance(image, torch.Tensor):
            raise ValueError(f"image must be a ComfyUI IMAGE Tensor [B,H,W,C], got {type(image)}")
        if image.ndim == 3:
            image = image.unsqueeze(0)
        if image.ndim != 4:
            raise ValueError(f"Bad image dims, expected [B,H,W,C], got {tuple(image.shape)}")

        # Neither dropdown applies in Custom mode. When film_type changes but
        # preset still sits on the previous category's stock, fall back to that
        # category's first entry, otherwise the stale combo value makes film_type
        # look like it did nothing.
        if mode == "Custom":
            preset = None
        elif _PRESET_TYPE.get(preset) != film_type:
            preset = next(n for n in PRESET_ORDER if _PRESET_TYPE[n] == film_type)

        params = _resolve_params(
            preset,
            preset_gain,
            {
                "grain_amount": grain_amount,
                "grain_size": grain_size,
                "grain_shadows": grain_shadows,
                "grain_chroma": grain_chroma,
                "halation": halation,
                "halation_threshold": halation_threshold,
                "halation_radius": halation_radius,
                "vignette": vignette,
                "vignette_size": vignette_size,
                "tone": tone,
                "micro_contrast": micro_contrast,
                "chroma_shift": chroma_shift,
                "split_tone": split_tone,
            },
        )

        # Process frame by frame: a 4K batch never has to hold the whole batch of
        # full-resolution noise tensors at once, and each frame gets independent noise
        frames = []
        for index in range(image.shape[0]):
            frame = image[index]
            alpha = frame[..., 3:4] if frame.shape[-1] == 4 else None
            rgb = frame[..., :3] if alpha is not None else frame
            out = film_finish(rgb, params, int(seed) + index)
            frames.append(torch.cat([out, alpha], dim=-1) if alpha is not None else out)

        return (torch.stack(frames).clamp(0, 1).to(image.dtype).contiguous(),)


NODE_CLASS_MAPPINGS = {"FeiFeiFilmGrainTone": FeiFeiFilmGrainTone}

NODE_DISPLAY_NAME_MAPPINGS = {"FeiFeiFilmGrainTone": "FeiFei Film Grain & Tone"}
