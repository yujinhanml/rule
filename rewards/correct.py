# --- 环境导入 ---
import sys
sys.path.append('/cpfs04/user/hanyujin/rule-gen/rule_tokenizer')
import torch
from modelling.sit import SiT_models
from modelling.tokenizer import SoftVQModel
from modelling.samplers import euler_sampler
from utils.model import build_tokenizer
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import ruamel.yaml as yaml
# --- 参数配置 ---
import numpy as np
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
from modelling.samplers import euler_sampler
from qwen_vl_utils import process_vision_info
import ruamel.yaml as yaml
from qwen_vl_utils import process_vision_info
import os
import torch
import torchvision.transforms.functional as TF
import torch.nn.functional as F
import os
import torch
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
from PIL import Image
import gc
import re


def samples_con(con, vq_mean, vq_std):
    B, N, C = con.shape
    H = W = int(N**0.5)
    con = con.permute(0, 2, 1).view(B, C, H, W)
    con = con * vq_std + vq_mean
    B, C, H, W = con.shape
    con = con.view(B, C, H * W).permute(0, 2, 1)
    return con

def correct_condition(image_input, device,processor, mllm_model,MLP, vq_mean, vq_std, image_size=128):
    """
    输入 PIL Image，通过 MLLM 获取目标数字，再用 CLIP 计算图文相似性，反向传播获得梯度。
    输出：(1, 3, H, W) 形状的 gradient map。
    """
    import torch.nn.functional as F
    import torchvision.transforms.functional as TF
    import re

    # Step 1: 构建带修改建议的 prompt
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_input, "resized_height": image_size, "resized_width": image_size},
                {"type": "text", "text": (
                    "You are an expert in image understanding. You will be shown images featuring four digits arranged in a grid from top to bottom and left to right. "
                    "First, extract the four digits in the order from top to bottom, then left to right. "
                    "Next, calculate the difference between each pair of adjacent digits, by subtracting the left digit from the right digit. "
                    "If all these differences are equal to 1, then the digits form an arithmetic sequence with a common difference of 1. "
                    "Score 0 — The digits do not form an arithmetic sequence with a difference of 1 between adjacent digits, or the image has fewer than four digits or contains duplicates. "
                    "Score 1 — The digits form a perfect arithmetic sequence with a difference of 1 between adjacent digits. "
                    "Please answer in the format: The reason is {{your_reason}}. The score is {{your_score}}."
                    "If the score is 0, also output: The correct numbers should be {{n1,n2,n3,n4}}."
                )}
            ]
        }
    ]
    # messages = [
    #     {
    #         "role": "user",
    #         "content": [
    #             {"type": "image", "image": image_input, "resized_height": image_size, "resized_width": image_size},
    #             {"type": "text", "text": (
    #                 "You are an expert in image understanding. You will be shown images featuring four digits arranged in a grid from top to bottom and left to right. "
    #                 "First, extract the four digits in the order from top to bottom, then left to right. "
    #                 "Next, calculate the difference between each pair of adjacent digits, by subtracting the left digit from the right digit. "
    #                 "If all these differences are equal to 2, then the digits form an arithmetic sequence with a common difference of 2. "
    #                 "Score 0 — The digits do not form an arithmetic sequence with a difference of 2 between adjacent digits, or the image has fewer than four digits or contains duplicates. "
    #                 "Score 1 — The digits form a perfect arithmetic sequence with a difference of 2 between adjacent digits. "
    #                 "Please first extract the digits in the format: The digits are {{n1,n2,n3,n4}}. Then give the score as: The score is {{score}}. "
    #                 "If the score is 0, also output: The correct numbers should be {{n1,n2,n3,n4}}."
    #             )}
    #         ]
    #     }
    # ]

    # Step 2: 发送给 MLLM
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        return_tensors="pt",
        padding=True
    ).to(device)

    with torch.no_grad():
        response = mllm_model.generate(**inputs, max_new_tokens=800)
        decoded = processor.batch_decode(response, skip_special_tokens=True)[0]
        # print(f"[MLLM Response] {decoded}")

    # Step 3: 提取目标数字和评分
    score_match = re.search(r'The score is\s*([01])', decoded)  # ✅ 简化处理
    # pred_digits_match = re.findall(r'The digits are\s*[\{\[]*([\d,\s]+)[\}\]]*', decoded)
    # pred_digits = pred_digits_match[-1].strip()
    score_match = re.search(r'The score is\s*\{{0,2}(\d+)\}{0,2}', decoded)
    correct_digits_match = re.findall(r'The correct numbers should be\s*[\{\[]*([\d,\s]+)[\}\]]*', decoded)
    correct_digits = correct_digits_match[-1].strip().rstrip('.')

    if not score_match:
        print("⚠️ 无法提取 score 或 prediction,返回零梯度")
        return None,score

    score = int(score_match.group(1))
    if score == 1:
        print(f"[MLLM] Score = {score}，图像已正确，无需梯度")
        return None,score

    if not correct_digits_match:
        print("⚠️ Score=0 但未能提取 correct digits,返回零梯度")
        return None,score
    # correct_text = "An image of digits 0,1,2,3"
    correct_text = f"An image of digits {correct_digits}"
    # f"An image of digits {correct_digits.replace(' ', '')}"
    # print(f"The correct label {correct_text}")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_input, "resized_height": image_size, "resized_width": image_size},
                {"type": "text", "text": (
                    correct_text
                )}
            ]
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs,
                        return_tensors="pt", padding=True).to(device)

    with torch.no_grad():
        outputs = mllm_model(**inputs, output_hidden_states=True, return_dict=True)
    hidden_states =  outputs.hidden_states[-1][:, -1, :].to(torch.float32)  # [B, T, D]
    # print("hidden_states:",hidden_states.shape)
    MLP = MLP.to(device)
    project_latents = MLP(hidden_states)  # [B, 32, 16, 16]
    project_latents = samples_con(project_latents, vq_mean, vq_std)
    return project_latents.to(device),score
    # if hidden_states.shape[1] == 37:
    #     project_latents = MLP(hidden_states)  # [B, 32, 16, 16]
    #     project_latents = samples_con(project_latents, vq_mean, vq_std)
    #     return project_latents.to(device),score
    # else:
    #     return None, score
