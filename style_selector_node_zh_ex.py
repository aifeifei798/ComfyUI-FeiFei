# -----------------------------------------------------------------
# 风格选择器扩展版（节点逻辑；模板数据已拆分到 style_data.py / juese_data.py）
# -----------------------------------------------------------------
import random
import time
import re

from .style_data import style_list
from .juese_data import juese_list


class StyleSelectorNodeZhex:
    """
    修复后的自定义节点
    """

    # 防止外部列表为空导致报错，加个判断或默认值
    style_names = [s["name"] for s in style_list] if "style_list" in globals() else []
    juese_names = [j["name"] for j in juese_list] if "juese_list" in globals() else []

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt1": (
                    "STRING",
                    {"multiline": True, "default": "A cute Japanese idol girl, portrait, masterpiece, best quality"},
                ),
                "prompt2": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "Underwater photography style (shooting through glass), twin tail girl submerged in water, cheeks puffed out blowing bubbles, twin tails floating freely in the water, dreamy and soft, cute expression.",
                    },
                ),
                "prompt3": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "Japanese gravure idol photography, realistic human, Fujifilm Provia color grading, soft pastel tones, high key lighting, clear skin texture, cinematic bokeh, 8k, highly detailed, natural skin, gorgeous, sharp focus, masterpiece, best quality.",
                    },
                ),
                # 注意：如果列表为空，ComfyUI可能会报错，建议确保 list 不为空
                "style_name": (cls.style_names,),
                "juese_names": (cls.juese_names,),
                "random_style": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("positive_prompt", "negative_prompt")
    FUNCTION = "apply_style"
    CATEGORY = "FeiFei"

    @classmethod
    def IS_CHANGED(
        cls, prompt1, prompt2, prompt3, style_name, juese_names, random_style
    ):
        # random 开启时强制刷新；关闭时返回 None 交给 ComfyUI 按输入哈希缓存
        # 注意：之前返回 float("NaN") 会因 NaN != NaN 导致永远判脏、缓存永不命中
        if random_style:
            return time.time_ns()
        return None

    def apply_style(
        self, prompt1, prompt2, prompt3, style_name, juese_names, random_style
    ):
        # 1. 拼接基础提示词
        # 加上空格防止粘连
        prompt = f"{prompt1} {prompt2} {prompt3}".strip()

        # 定义初始的负面提示词
        current_negative = ""
        # ===========================
        # 第一步：处理角色 (Juese)
        # 逻辑：先找到角色，把角色的 prompt 追加到你的输入 prompt 后面
        # ===========================
        selected_juese = next((j for j in juese_list if j["name"] == juese_names), None)

        if (
            selected_juese and selected_juese["name"] != "(None)"
        ):  # 假设你有一个选项叫 (None)
            # 获取角色提示词，如果没有则为空
            juese_prompt = selected_juese.get("prompt", "")
            juese_negative = selected_juese.get("negative_prompt", "")

            # 将角色词追加到主提示词
            prompt = f"{prompt}, {juese_prompt}"
            current_negative = f"{current_negative}, {juese_negative}"
        # ===========================
        # 第二步：处理风格 (Style)
        # 逻辑：将处理过角色的 prompt，填入风格的模板中
        # ===========================
        selected_style = None
        if random_style:
            # 过滤掉无效风格
            eligible_styles = [s for s in style_list if s["name"] != "(None)"]
            if eligible_styles:
                selected_style = random.choice(eligible_styles)
                print(
                    f"[StyleSelector] Random style selected: {selected_style['name']}"
                )
            else:
                # 如果没有可选风格，回落到当前选择
                selected_style = next(
                    (s for s in style_list if s["name"] == style_name), None
                )
        else:
            selected_style = next(
                (s for s in style_list if s["name"] == style_name), None
            )
        # 应用风格模板
        if selected_style and selected_style["name"] != "(None)":
            prompt_template = selected_style.get("prompt", "{prompt}")
            negative_template = selected_style.get("negative_prompt", "")
            # 替换模板中的 {prompt}
            final_positive = prompt_template.replace("{prompt}", prompt)

            # 处理负面提示词
            # 这里的逻辑看你需求：通常风格的负面提示词是固定的，或者追加
            # 假设风格负面提示词模板里也有 {prompt}（虽然少见），如果没有，就直接拼接
            if "{prompt}" in negative_template:
                final_negative = negative_template.replace("{prompt}", current_negative)
            else:
                # 如果模板里没 {prompt}，通常意味着风格自带通用负面词，我们把之前的角色负面词加上去
                final_negative = f"{negative_template}, {current_negative}".strip(", ")
        else:
            # 如果没选风格，就直接输出当前结果
            final_positive = prompt
            final_negative = current_negative
        # 清理一下多余的逗号和空格
        final_positive = final_positive.strip(", ")
        final_positive = re.sub(r",\s*,", ",", final_positive)
        final_positive = re.sub(r"\s+", " ", final_positive)
        final_positive = (
            final_positive.replace(", ,", ",").replace(",,", ",").strip(", ")
        )
        final_negative = final_negative.strip(", ")
        print(final_positive)
        return (final_positive, final_negative)


# -----------------------------------------------------------------
#  ComfyUI 必须的映射字典
#  这告诉ComfyUI如何加载和显示这个节点
# -----------------------------------------------------------------
NODE_CLASS_MAPPINGS = {"StyleSelectorNodeZhex": StyleSelectorNodeZhex}
NODE_DISPLAY_NAME_MAPPINGS = {"StyleSelectorNodeZhex": "Style Selector EX"}
