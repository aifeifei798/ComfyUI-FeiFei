import torch


class ImageToRGB:
    """
    强制将输入的任何图像 Tensor 转为合法的 3 通道 (RGB) 格式，
    并保证连续内存 (contiguous)，完美兼容 NVIDIA RTX VSR / nvvfx。
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
    CATEGORY = "image/color"

    def convert_to_rgb(self, image: torch.Tensor):
        if not isinstance(image, torch.Tensor):
            return (image, )

        # 1. 确保是 4 维张量 [B, H, W, C]
        if image.ndim == 3:
            image = image.unsqueeze(0)

        # 2. 如果通道在前 (NCHW -> NHWC)
        if image.ndim == 4 and image.shape[1] in [1, 3, 4
                                                  ] and image.shape[-1] > 4:
            image = image.permute(0, 2, 3, 1)

        channels = image.shape[-1]

        # 3. 核心：强制转换为 3 通道 (RGB)
        if channels == 4:
            # RGBA -> 截取前 3 通道 RGB（丢弃 Alpha 透明通道）
            image = image[..., :3]
        elif channels == 1:
            # 灰度图 -> 复制 3 份扩展为 RGB
            image = image.repeat(1, 1, 1, 3)
        elif channels > 4:
            image = image[..., :3]
        elif channels == 2:
            # 极特殊双通道补齐
            pad = torch.zeros_like(image[..., :1])
            image = torch.cat([image, pad], dim=-1)

        # 4. 关键：保证内存连续！NVIDIA RTX VSR 底层通过 C++ 指针/DLPack 读取，必须 contiguous
        image = image.contiguous()

        return (image, )


# 注册节点到 ComfyUI
NODE_CLASS_MAPPINGS = {"ImageToRGB": ImageToRGB}

NODE_DISPLAY_NAME_MAPPINGS = {"ImageToRGB": "Image To RGB (Force 3-Channel)"}
