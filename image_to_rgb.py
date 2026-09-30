import torch


class ImageToRGB:
    """
    Force any incoming image tensor into a valid 3-channel RGB layout and make
    it contiguous, so NVIDIA RTX VSR / nvvfx can consume it.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", ),
            }
        }

    RETURN_TYPES = ("IMAGE", )
    RETURN_NAMES = ("image", )
    FUNCTION = "convert_to_rgb"
    CATEGORY = "FeiFei"

    def convert_to_rgb(self, image: torch.Tensor):
        if not isinstance(image, torch.Tensor):
            raise TypeError(f"image must be a torch.Tensor, got {type(image)}")

        # 1. Normalize to 4 dims [B, H, W, C]
        if image.ndim == 2:
            # Single grayscale [H, W] -> [1, H, W, 1]
            image = image.unsqueeze(0).unsqueeze(-1)
        elif image.ndim == 3:
            # Could be [H, W, C] or an unbatched [C, H, W]; treat as [H, W, C] first
            # The [C, H, W] case is detected as NCHW and permuted in step 2
            image = image.unsqueeze(0)
        if image.ndim != 4:
            raise ValueError(f"Bad image dims, expected [B,H,W,C], got {tuple(image.shape)}")

        # 2. Channels first (NCHW -> NHWC); ambiguous square images stay BHWC
        c1 = image.shape[1]
        c_last = image.shape[-1]
        if c1 in (1, 2, 3, 4) and c_last not in (1, 2, 3, 4):
            image = image.permute(0, 2, 3, 1)

        channels = image.shape[-1]
        if channels <= 0:
            raise ValueError(f"Bad channel count: {channels}")

        # 3. Core: force 3 channels (RGB)
        if channels == 4:
            # RGBA -> keep the first 3 channels (drop alpha)
            image = image[..., :3]
        elif channels == 1:
            # Grayscale -> repeat 3 times to expand to RGB
            image = image.repeat(1, 1, 1, 3)
        elif channels > 4:
            image = image[..., :3]
        elif channels == 2:
            # Extremely rare 2-channel case, pad up to 3
            pad = torch.zeros_like(image[..., :1])
            image = torch.cat([image, pad], dim=-1)

        # 4. Contiguity matters: NVIDIA RTX VSR reads the buffer through raw C++ pointers / DLPack
        image = image.contiguous()

        return (image, )


# Register the node with ComfyUI
NODE_CLASS_MAPPINGS = {"ImageToRGB": ImageToRGB}

NODE_DISPLAY_NAME_MAPPINGS = {"ImageToRGB": "Image To RGB (Force 3-Channel)"}
