import os
import json
import re
import math
import urllib.request
import urllib.error

# 常见比例对应的推荐分辨率 (以 ~1.5M/2K 像素为基准，对齐 16 的倍数)
RATIO_MAP_2K = {
    "1:1": (1536, 1536),
    "3:2": (1872, 1248),
    "2:3": (1248, 1872),
    "16:9": (2016, 1152),
    "9:16": (1152, 2016),
    "4:3": (1728, 1296),
    "3:4": (1296, 1728),
    "21:9": (2304, 992),
    "2:1": (2048, 1024),
    "7:3": (2352, 1008),
    "18:39": (1056, 2288),
    "9:20": (1088, 2416),
    "5:7": (1280, 1792),
    "7:5": (1792, 1280),
}

def parse_wh_ratio(ratio_str, target_pixel_count=1536*1536):
    """根据宽高比计算宽和高（16的倍数）"""
    ratio_str = ratio_str.strip()
    if ratio_str in RATIO_MAP_2K:
        return RATIO_MAP_2K[ratio_str]
    
    # 解析自定义比例如 "16:9"
    match = re.match(r"(\d+(?:\.\d+)?)\s*[:：/]\s*(\d+(?:\.\d+)?)", ratio_str)
    if match:
        w_factor = float(match.group(1))
        h_factor = float(match.group(2))
        if h_factor > 0:
            ratio = w_factor / h_factor
            height = math.sqrt(target_pixel_count / ratio)
            width = height * ratio
            # 对齐到 16 的倍数
            width = int(round(width / 16.0) * 16)
            height = int(round(height / 16.0) * 16)
            return width, height
            
    # 默认 1:1
    return 1536, 1536

class QwenImagePromptEnhancer:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "user_prompt": ("STRING", {"multiline": True, "default": "Tokyo Japanese girl walking in the rain with umbrella"}),
                "mode": (["T2I (文生图)", "I2I (图生图)"], {"default": "T2I (文生图)"}),
                "api_base": ("STRING", {"default": "http://127.0.0.1:8080"}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.1, "max": 1.5, "step": 0.05}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "INT", "INT", "STRING")
    RETURN_NAMES = ("rewritten_prompt", "wh_ratio", "width", "height", "ratio_follow")
    FUNCTION = "enhance_prompt"
    CATEGORY = "FeiFei"

    def load_system_prompt(self, mode):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        if "T2I" in mode:
            filename = "Qwen-Image-2.1-T2I.system_prompt.txt"
        else:
            filename = "Qwen-Image-2.1-I2I.system_prompt.txt"
            
        filepath = os.path.join(current_dir, filename)
        if os.path.exists(filepath):
            with open(filepath, "r", encoding="utf-8") as f:
                return f.read()
        return "You are an expert at enhancing image prompts. Output valid JSON."

    def enhance_prompt(self, user_prompt, mode, api_base, temperature):
        system_prompt = self.load_system_prompt(mode)
        url = api_base.rstrip("/") + "/v1/chat/completions"

        payload = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": temperature,
            "stream": False
        }

        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})

        raw_content = ""
        try:
            with urllib.request.urlopen(req, timeout=120) as response:
                res_json = json.loads(response.read().decode("utf-8"))
                raw_content = res_json["choices"][0]["message"]["content"].strip()
        except urllib.error.URLError as e:
            # 兼容 llama.cpp 原生 /completion 接口
            raw_url = api_base.rstrip("/") + "/completion"
            raw_payload = {
                "prompt": f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n",
                "temperature": temperature,
                "n_predict": 1024
            }
            try:
                req_raw = urllib.request.Request(raw_url, data=json.dumps(raw_payload).encode("utf-8"), headers={"Content-Type": "application/json"})
                with urllib.request.urlopen(req_raw, timeout=120) as resp:
                    raw_content = json.loads(resp.read().decode("utf-8")).get("content", "").strip()
            except Exception as ex:
                return (f"API Error: {str(e)} / {str(ex)}", "", 1024, 1024, "")

        # 解析 LLM 返回的 JSON
        rewritten_prompt = raw_content
        wh_ratio = "3:2" if "T2I" in mode else ""
        ratio_follow = ""

        try:
            # 提取可能夹在 markdown 代码块里的 JSON
            match = re.search(r"\{.*\}", raw_content, re.DOTALL)
            if match:
                parsed = json.loads(match.group(0))
                rewritten_prompt = parsed.get("rewritten_prompt", raw_content)
                wh_ratio = parsed.get("wh_ratio", "")
                ratio_follow = parsed.get("ratio_follow", "")
        except Exception:
            # 如果模型没有严格输出 json，则退回原始文本
            pass

        # 计算尺寸
        width, height = parse_wh_ratio(wh_ratio if wh_ratio else "1:1")

        return (rewritten_prompt, wh_ratio, width, height, ratio_follow)