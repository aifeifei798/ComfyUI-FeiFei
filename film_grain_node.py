"""物理胶片与呼吸感后处理：镜头 → 胶片基 → 冲印 → 乳剂，四段流水线。

纯 torch 实现，不依赖任何外部模型或大体积库。处理顺序按真实的胶片流程排：
镜头（暗角 / 侧向色散）→ 胶片基光晕（halation）→ 冲印 S 曲线与分离色调 →
微对比 → 乳剂颗粒。颗粒放最后，避免被冲印曲线压掉。
"""

import math

import torch
import torch.nn.functional as F

# 所有「尺寸性格」以 1024 短边为基准，颗粒、光晕、微对比随画幅自动缩放
RES_REF = 1024.0

# 颗粒强度换算：默认 grain_amount=0.25 约 ±3.5/255（真实扫描的量级），
# 拉满 1.0 约 ±14/255。超过这个就开始像电视雪花而不是胶片了。
GRAIN_GAIN = 0.10
# 颗粒团里逐像素细节的占比。太高会退化成椒盐噪点，保持很低才像成团的银盐颗粒
GRAIN_FINE_MIX = 0.06
# 1024 短边下 grain_size=1.0 对应约 2.5px 的颗粒团，跟真实胶片扫描接近
GRAIN_CELL_SCALE = 2.5

LUMA_WEIGHTS = (0.2126, 0.7152, 0.0722)
HALATION_TINT = (1.0, 0.62, 0.42)

# 绝大多数胶片共用的性格底子，具体的胶片只覆盖自己不一样的项
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

# 第一个下拉：胶片大类。Digital Clean / 过期片这类不是「某一卷胶片」，
# 单独归到 Other，不跟真实胶片混在一起。UI 上可见的值一律英文。
FILM_TYPES = [
    "Color Negative",
    "Slide",
    "B&W Negative",
    "Cinema",
    "Special",
    "Other",
]

# 第二个下拉按大类分组，顺序即下拉顺序。每种胶片只写与 _BASE_TRAITS 的差异项。
FILM_PRESETS = {
    # ── Color Negative ── 日常人像与街拍
    "Portra 160": {"grain_amount": 0.12, "grain_size": 0.6, "grain_shadows": 0.50, "tone": 0.14, "micro_contrast": 0.08},
    "Portra 400": {"grain_amount": 0.16, "grain_size": 0.7, "grain_shadows": 0.50, "tone": 0.18, "micro_contrast": 0.10, "halation_threshold": 0.82, "halation_radius": 0.8, "vignette_size": 0.75, "chroma_shift": 0.8},
    "Portra 800": {"grain_amount": 0.24, "grain_size": 0.95, "grain_shadows": 0.55, "tone": 0.19, "micro_contrast": 0.11, "halation_threshold": 0.82, "halation_radius": 0.8, "vignette_size": 0.75},
    "Ektar 100": {"grain_amount": 0.14, "grain_size": 0.6, "grain_shadows": 0.45, "tone": 0.30, "micro_contrast": 0.22, "split_tone": 0.03},
    "Ektar 500": {"grain_amount": 0.28, "grain_size": 1.0, "grain_shadows": 0.50, "tone": 0.32, "micro_contrast": 0.22, "split_tone": 0.03},
    "Gold 200": {"grain_amount": 0.24, "grain_size": 1.1, "grain_shadows": 0.55, "tone": 0.28, "micro_contrast": 0.18, "split_tone": 0.07},
    "Gold 400": {"grain_amount": 0.30, "grain_size": 1.3, "grain_shadows": 0.58, "tone": 0.30, "micro_contrast": 0.19, "split_tone": 0.08},
    "Fuji C200": {"grain_amount": 0.22, "grain_size": 1.0, "grain_shadows": 0.52, "tone": 0.34, "micro_contrast": 0.20, "split_tone": 0.06},
    "Pro 400H": {"grain_amount": 0.24, "grain_size": 1.0, "grain_shadows": 0.62, "tone": 0.30, "micro_contrast": 0.20, "halation_tint": (0.90, 0.96, 1.00), "split_tone": 0.10},

    # ── Slide ── 反转片颗粒极细、反差极大
    "Ektachrome E100": {"grain_amount": 0.05, "grain_size": 0.4, "grain_shadows": 0.40, "grain_chroma": 0.10, "tone": 0.46, "micro_contrast": 0.28, "halation": 0.05, "halation_threshold": 0.86, "halation_radius": 0.8, "vignette": 0.12, "split_tone": 0.03},
    "Fuji Provia 100F": {"grain_amount": 0.07, "grain_size": 0.5, "grain_shadows": 0.42, "grain_chroma": 0.12, "tone": 0.42, "micro_contrast": 0.26, "halation": 0.06, "split_tone": 0.03},

    # ── B&W Negative ── 颗粒是主角，彩色颗粒压到最低
    "Tri-X 400": {"grain_amount": 0.38, "grain_size": 1.8, "grain_shadows": 0.72, "grain_chroma": 0.12, "tone": 0.35, "micro_contrast": 0.22, "halation": 0.05, "halation_threshold": 0.86, "halation_radius": 0.7, "vignette": 0.06, "vignette_size": 0.80, "chroma_shift": 1.2, "split_tone": 0.03},
    "HP5 Plus 400": {"grain_amount": 0.34, "grain_size": 1.4, "grain_shadows": 0.68, "grain_chroma": 0.10, "tone": 0.30, "micro_contrast": 0.20, "halation": 0.05, "vignette": 0.07, "chroma_shift": 1.0, "split_tone": 0.03},
    "FP4 Plus 125": {"grain_amount": 0.30, "grain_size": 1.25, "grain_shadows": 0.64, "grain_chroma": 0.10, "tone": 0.28, "micro_contrast": 0.18, "halation": 0.05, "split_tone": 0.04},
    "TMax 400": {"grain_amount": 0.20, "grain_size": 0.6, "grain_shadows": 0.55, "grain_chroma": 0.08, "tone": 0.32, "micro_contrast": 0.24, "halation": 0.04},
    "TMax 100": {"grain_amount": 0.16, "grain_size": 0.5, "grain_shadows": 0.52, "grain_chroma": 0.08, "tone": 0.28, "micro_contrast": 0.22, "halation": 0.04},
    "Delta 3200": {"grain_amount": 0.18, "grain_size": 0.55, "grain_shadows": 0.58, "grain_chroma": 0.10, "tone": 0.30, "micro_contrast": 0.22, "halation": 0.04},
    "Delta 100": {"grain_amount": 0.14, "grain_size": 0.45, "grain_shadows": 0.50, "grain_chroma": 0.08, "tone": 0.26, "micro_contrast": 0.20, "halation": 0.04},
    "Acros 100": {"grain_amount": 0.15, "grain_size": 0.5, "grain_shadows": 0.50, "grain_chroma": 0.08, "tone": 0.28, "micro_contrast": 0.20, "halation": 0.04, "split_tone": 0.04},

    # ── Cinema ──  CineStill 的红光晕是主角
    "CineStill 800T": {"grain_amount": 0.22, "grain_size": 1.2, "grain_shadows": 0.60, "grain_chroma": 0.30, "tone": 0.28, "micro_contrast": 0.12, "halation": 0.34, "halation_threshold": 0.70, "halation_radius": 1.8, "halation_tint": (1.0, 0.42, 0.34), "vignette": 0.16, "vignette_size": 0.70, "chroma_shift": 1.0, "split_tone": 0.09},
    "CineStill 400D": {"grain_amount": 0.20, "grain_size": 1.0, "grain_shadows": 0.58, "tone": 0.26, "micro_contrast": 0.14, "halation": 0.26, "halation_threshold": 0.72, "halation_radius": 1.5, "vignette": 0.14, "vignette_size": 0.70, "split_tone": 0.08},
    "Vision3 250D": {"grain_amount": 0.18, "grain_size": 0.8, "grain_shadows": 0.50, "tone": 0.24, "micro_contrast": 0.16, "halation": 0.08},
    "Vision3 500T": {"grain_amount": 0.28, "grain_size": 1.2, "grain_shadows": 0.58, "tone": 0.28, "micro_contrast": 0.18, "halation": 0.12, "vignette": 0.12},
    "Cine 50D": {"grain_amount": 0.16, "grain_size": 0.7, "grain_shadows": 0.48, "tone": 0.22, "micro_contrast": 0.14, "halation": 0.08},
    "500T Expired": {"grain_amount": 0.40, "grain_size": 1.9, "grain_shadows": 0.68, "tone": 0.34, "micro_contrast": 0.16, "halation": 0.22, "halation_threshold": 0.74, "halation_radius": 1.5, "halation_tint": (1.0, 0.56, 0.46), "vignette": 0.14, "split_tone": 0.10},

    # ── Special ── 强味道的效果，不是标准胶片
    "Cinestack 800T": {"grain_amount": 0.20, "grain_size": 1.1, "halation": 0.60, "halation_threshold": 0.55, "halation_radius": 2.6, "vignette": 0.20, "split_tone": 0.12},
    "Push +2 Stops": {"grain_amount": 0.52, "grain_size": 2.2, "grain_shadows": 0.72, "tone": 0.42, "micro_contrast": 0.24, "halation": 0.14, "vignette": 0.14},
    "Cross Process": {"grain_amount": 0.26, "grain_size": 1.0, "tone": 0.40, "micro_contrast": 0.26, "halation": 0.18, "halation_tint": (1.0, 0.52, 0.60), "split_tone": 0.12},

    # ── Other ── 参照组与数字味，方便 A/B 对比
    "custom": {},
    "Digital Clean": {"grain_amount": 0.05, "grain_size": 0.6, "grain_shadows": 0.40, "grain_chroma": 0.15, "tone": 0.10, "micro_contrast": 0.06, "halation": 0.05, "halation_threshold": 0.85, "halation_radius": 0.8, "vignette": 0.04, "vignette_size": 0.80, "chroma_shift": 0.4, "split_tone": 0.02},
}

# 每种胶片在第一个下拉里属于哪一类
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
    "custom": "Other", "Digital Clean": "Other",
}

# 第二个下拉的可选值：把整张表按 FILM_TYPES 的顺序重排一次
PRESET_ORDER = [
    name for film_type in FILM_TYPES for name, kind in _PRESET_TYPE.items() if kind == film_type
]


def _resolve_params(preset, mix, sliders):
    """把预设按 mix 混进滑杆值，纯函数便于单测。

    mix=0 或 preset=custom 时原样返回滑杆值，所以滑杆永远真实生效，
    不会出现「选了预设再拖滑杆没反应」的骗 UI 情况。
    """
    params = dict(sliders, halation_tint=HALATION_TINT)
    stock = FILM_PRESETS.get(preset)
    # custom 是「不套预设」，不能落到共享底子上 —— 它的覆盖是空的
    if stock is None or not stock or mix <= 0.0:
        return params
    target = {**_BASE_TRAITS, **stock}
    for key, value in target.items():
        current = params[key]
        if isinstance(value, tuple):
            params[key] = tuple(c + (v - c) * mix for c, v in zip(current, value))
        else:
            params[key] = current + (value - current) * mix
    return params


def _luma(img):
    """Rec.709 亮度，通道数不足时补零再归一化"""
    channels = img.shape[-1]
    weights = list(LUMA_WEIGHTS[:channels])
    weights += [0.0] * (channels - len(weights))
    w = torch.tensor(weights, device=img.device, dtype=img.dtype)
    return (img * (w / w.sum())).sum(-1, keepdim=True)


def _radial(height, width, device, dtype):
    """归一化半径：画面半宽/半高处为 1，四角约 1.41"""
    y = torch.linspace(-1.0, 1.0, height, device=device, dtype=dtype).view(1, height, 1)
    x = torch.linspace(-1.0, 1.0, width, device=device, dtype=dtype).view(1, 1, width)
    return (x * x + y * y).sqrt()


def _blur(plane, sigma):
    """可分离高斯，反射填充，plane: [B,C,H,W]"""
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
    """size 是衰减起点占半径的比例；四角最多压到 1-amount，不会压成纯黑"""
    falloff = ((radial / size - 1.0) / 0.4142).clamp(0, 1) ** 1.8
    return img * (1.0 - amount * falloff).unsqueeze(-1)


def _chroma_shift(img, radial, amount_px):
    """红蓝通道按半径向外多放大一点、绿通道不动 —— 侧向色散，中心与边缘中点几乎无感"""
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
    """高光穿过胶片基散射成的暖色光晕：降分辨率模糊后 screen 混合"""
    height, width = img.shape[1:3]
    mask = ((_luma(img) - threshold) / max(1e-3, 1.0 - threshold)).clamp(0, 1) ** 1.5
    sigma = radius * min(height, width) / RES_REF * 8.0
    # 光晕是低频现象：降到 sigma≈6 的工作分辨率再模糊，4K 大半径也不会变慢
    down = min(16, max(2, int(round(sigma / 6.0))))
    work = mask.permute(0, 3, 1, 2)
    small = F.interpolate(work, size=(max(1, height // down), max(1, width // down)), mode="area")
    glow = _blur(small, sigma / down)
    glow = F.interpolate(glow, size=(height, width), mode="bilinear", align_corners=False)
    tint_t = torch.tensor(list(tint)[:img.shape[-1]], device=img.device, dtype=img.dtype)
    return 1.0 - (1.0 - img) * (1.0 - glow.permute(0, 2, 3, 1) * tint_t * amount)


def _tone(img, amount, curve=0.35, lift=0.015, mid=0.18):
    """冲印 S 曲线：在 log 密度域以中灰为轴展开中调、压缩暗部与亮部。

    正弦项在 ±L 处斜率回落，两端严格守恒 —— 纯白仍是 1.0、中灰不变，
    所以高光只会被压肩不会被推出画面（多项式 S 曲线做不到这点）。
    """
    stops = -math.log(mid)
    ld = (torch.log(_luma(img).clamp(1e-4, 1.0)) - math.log(mid)).clamp(-stops, stops)
    shaped = ld + curve * (stops / math.pi) * torch.sin(math.pi * ld / stops)
    graded = img * torch.exp(shaped - ld) + lift * (1.0 - img)
    return img + (graded - img) * amount


def _split_tone(img, amount):
    """暗部偏暖、高光偏冷，量很小，只是让颜色不那么数字味"""
    luma = _luma(img).clamp(0.0, 1.0)
    channels = img.shape[-1]
    warm = torch.tensor([1.06, 1.0, 0.94][:channels], device=img.device, dtype=img.dtype)
    cool = torch.tensor([0.97, 1.0, 1.05][:channels], device=img.device, dtype=img.dtype)
    return img * (1.0 + ((warm * (1.0 - luma) + cool * luma) - 1.0) * amount)


def _micro_contrast(img, amount):
    """小半径 unsharp，只动亮度不加彩边，把皮肤和布料的蜡感找回来"""
    luma = _luma(img)
    sigma = 1.2 * min(img.shape[1], img.shape[2]) / RES_REF
    base = _blur(luma.permute(0, 3, 1, 2), sigma).permute(0, 2, 3, 1)
    return img + (luma - base) * amount


def _grain_field(height, width, channels, cell_px, seed, device, dtype):
    """低分辨率噪声场上采样得到的成团颗粒，混少量逐像素细节后归一化"""
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
    """颗粒强度按亮度加权：中调最强，向暗部按 shadows 倾斜，亮部快速衰减"""
    luma = _luma(img)
    up = (luma / 0.5).clamp(0, 1) ** (0.35 + (1.0 - shadows) * 1.3)
    down = ((1.0 - luma) / 0.5).clamp(0, 1) ** 1.6
    response = torch.where(luma < 0.5, up, down)

    height, width, channels = img.shape[1], img.shape[2], img.shape[3]
    cell = max(1.0, size * GRAIN_CELL_SCALE * min(height, width) / RES_REF)
    field = _grain_field(height, width, channels, cell, seed, img.device, img.dtype)
    if chroma < 1.0:
        # chroma=0 全单色，=1 三通道独立；解析补偿让总幅度不随 chroma 变化
        mono = field.mean(dim=-1, keepdim=True)
        field = (mono + (field - mono) * chroma) * (3.0 / (1.0 + 2.0 * chroma ** 2)) ** 0.5
    return img + field * response * (amount * GRAIN_GAIN)


def film_finish(frame, params, seed):
    """单帧处理，frame: [H,W,C] 浮点张量，纯函数便于单测"""
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
    物理胶片与呼吸感后处理：有机胶片颗粒 + 微弱 halation + 暗角色散 +
    冲印 S 曲线 + 微对比，给 AI 图像去掉「太干净太塑料」的观感。
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "film_type": (FILM_TYPES, {
                    "default": "Color Negative",
                    "tooltip": "Film family. Only groups the preset list below - it applies no effect of its own, "
                               "so pick any family and set preset = custom to hand-tune everything yourself.",
                }),
                "preset": (PRESET_ORDER, {
                    "default": "Portra 400",
                    "tooltip": "Film stock to imitate. Only grain size, grain response, halation and lens "
                               "falloff are changed - never a colour grade, so the image keeps its own colours. "
                               "Choose 'custom' to ignore all presets and drive every slider yourself.",
                }),
                "preset_mix": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "How much of the preset to keep. This is the second way to go fully manual: "
                               "1.0 = pure preset, sliders ignored. 0.5 = half preset, half your sliders. "
                               "0.0 = pure sliders, preset ignored. "
                               "Fully manual = preset_mix 0.0, or preset = custom (same result either way).",
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
                "micro_contrast": ("FLOAT", {
                    "default": 0.15, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Fine detail sharpening on luminance only, no colour fringes. "
                               "The single most effective slider against the waxy AI-skin look.",
                }),
                "chroma_shift": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 8.0, "step": 0.1,
                    "tooltip": "Lateral chromatic aberration, in pixels of fringing at the corner "
                               "of a 1024px image. Invisible in the centre.",
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
        film_type,
        preset,
        preset_mix,
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
        micro_contrast=0.15,
        chroma_shift=1.0,
        split_tone=0.06,
    ):
        if not isinstance(image, torch.Tensor):
            raise ValueError(f"image must be a ComfyUI IMAGE Tensor [B,H,W,C], got {type(image)}")
        if image.ndim == 3:
            image = image.unsqueeze(0)
        if image.ndim != 4:
            raise ValueError(f"Bad image dims, expected [B,H,W,C], got {tuple(image.shape)}")

        # 换了 film_type 但 preset 还停在上一个类别的胶片上时，回落到该类别的第一款，
        # 否则 combo 里残留的值会让 film_type 看起来没生效。custom 是「不套预设」的意思，
        # 和分类无关，任何 film_type 下都必须原样尊重。
        if preset != "custom" and _PRESET_TYPE.get(preset) != film_type:
            preset = next(n for n in PRESET_ORDER if _PRESET_TYPE[n] == film_type)

        params = _resolve_params(
            preset,
            preset_mix,
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

        # 逐帧处理：4K 批量时不必同时持有整批全分辨率噪点张量，顺带让每帧噪声独立
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
