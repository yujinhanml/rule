import re
import torch
import gc
import re
from qwen_vl_utils import process_vision_info
import os
import torch
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
from PIL import Image
import torch
import re
import gc
from tqdm import tqdm

# def mllm_score(image_input, image_size=128, mllm_model=None, processor=None):
#     """
#     用 Qwen2.5-VL 模型对图片进行评分，返回整数分数 [0,1,2,...,10]
#     """
#     try:
#         messages = [
#             {
#                 "role": "user",
#                 "content": [
#                     {
#                         "type": "image",
#                         "image": image_input,
#                         "resized_height": image_size,
#                         "resized_width": image_size
#                     },
#                     {
#                         "type": "text",
#                         "text": (
#                             "You are an expert in image understanding. You will be shown images featuring four digits arranged in a grid from top to bottom and left to right. "
#                             "First, extract the four digits in the order from top to bottom, then left to right (e.g., extract 0, 1, 2, 3). "
#                             "Next, calculate the difference between each pair of adjacent digits (i.e., check if 1 - 0 = 2 - 1 = 3 - 2 = 1). "
#                             "If the differences between all adjacent digits are equal to 1, then the digits form an arithmetic sequence with a common difference of 1, such as (0,1,2,3) and (5,6,7,8). "
#                             "Score 0 — The digits do not form an arithmetic sequence with a difference of 1 between adjacent digits, or the image has fewer than four digits or contains duplicates. "
#                             "Score 1 — The digits form a perfect arithmetic sequence with a difference of 1 between adjacent digits. "
#                             "Please answer in the format: The reason is {{your_reason}}. The score is {{your_score}}."
#                         )
#                     }
#                 ],
#             }
#         ]

#         # 构造输入
#         text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
#         image_inputs, video_inputs = process_vision_info(messages)
#         inputs = processor(
#             text=[text],
#             images=image_inputs,
#             videos=video_inputs,
#             padding=True,
#             return_tensors="pt"
#         ).to("cuda")

#         # 推理
#         with torch.no_grad():
#             generated_ids = mllm_model.generate(**inputs, max_new_tokens=512)
#             generated_ids_trimmed = [
#                 out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
#             ]
#             output_text = processor.batch_decode(
#                 generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
#             )

#         # print("output_text:", output_text)

#         # 正则匹配所有可能形式的分数
#         score_patterns = [
#             r"The score is\s*\{{0,2}(\d+)\}{0,2}",     # 支持 0-2 个大括号包裹的数字
#             r"\*\*Final Answer\*\*.*?(\d+)",
#         ]

#         for pattern in score_patterns:
#             match = re.search(pattern, output_text[0], re.DOTALL)
#             if match:
#                 return int(match.group(1))

#         print("⚠️ Could not parse score from model output.")
#         return 0  # fallback

#     except Exception as e:
#         print(f"⚠️ mllm_score error: {e}")
#         return 0  # fallback

#     finally:
#         torch.cuda.empty_cache()
#         gc.collect()




def mllm_score(
    image_list,
    image_size=128,
    mllm_model=None,
    processor=None,
    device="cuda",
    batch_size=8,  # 每次送入模型的图像数量
):
    """
    给一批 PIL 图像打分，返回 List[int] 分数，在 `generate()` 推理阶段显示进度
    """
    all_scores = []

    for i in tqdm(range(0, len(image_list), batch_size), desc="MLLM Scoring"):
        batch_images = image_list[i:i + batch_size]

        # 构造 batch prompt
        messages_list = [
            [{
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": img,
                        "resized_height": image_size,
                        "resized_width": image_size
                    },
                    {
                        "type": "text",
                        "text": (
                            "You are an expert in image understanding. You will be shown images featuring four digits arranged in a grid from top to bottom and left to right. "
                            "First, extract the four digits in the order from top to bottom, then left to right (e.g., extract 0, 1, 2, 3). "
                            "Next, calculate the difference between each pair of adjacent digits (i.e., check if 1 - 0 = 2 - 1 = 3 - 2 = 1). "
                            "If the differences between all adjacent digits are equal to 1, then the digits form an arithmetic sequence with a common difference of 1, such as (0,1,2,3) and (5,6,7,8). "
                            "Score 0 — The digits do not form an arithmetic sequence with a difference of 1 between adjacent digits, or the image has fewer than four digits or contains duplicates. "
                            "Score 1 — The digits form a perfect arithmetic sequence with a difference of 1 between adjacent digits. "
                            "Please answer in the format: The reason is {{your_reason}}. The score is {{your_score}}."
                        )
                    }
                ]
            }] for img in batch_images
        ]

        texts = [
            processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
            for msg in messages_list
        ]

        image_inputs, video_inputs = process_vision_info([m[0] for m in messages_list])

        inputs = processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
            # padding="max_length",
            # truncation=True,
            # max_length=512,
        ).to(device)

        try:
            with torch.no_grad():
                generated_ids = mllm_model.generate(**inputs, max_new_tokens=512)
                generated_ids_trimmed = [
                    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                output_texts = processor.batch_decode(
                    generated_ids_trimmed,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False
                )

            # === Parse scores ===
            score_patterns = [
                r"The score is\s*\{{0,2}(\d+)\}{0,2}",
                r"\*\*Final Answer\*\*.*?(\d+)",
            ]

            for text in output_texts:
                for pattern in score_patterns:
                    match = re.search(pattern, text, re.DOTALL)
                    if match:
                        all_scores.append(int(match.group(1)))
                        break
                else:
                    print(f"⚠️ No score parsed from text: {text}")
                    all_scores.append(0)

        except Exception as e:
            print(f"⚠️ MLLM batch error: {e}")
            all_scores.extend([0] * len(batch_images))

        finally:
            torch.cuda.empty_cache()
            gc.collect()

    return all_scores


def extract_score_after_final_answer(output_text):
    """
    从给定文本中查找 **Final Answer** 后面出现的第一个数字，并返回其整数值。
    如果未找到，则抛出 ValueError。
    """
    # 若 output_text 是列表，则默认取第一个元素
    if isinstance(output_text, list):
        text_to_search = output_text[0]
    else:
        text_to_search = output_text

    # 正则：匹配 "**Final Answer**" 后，任意字符(含换行)，再遇到数字(\d+)
    pattern = r"\*\*Final Answer\*\*.*?(\d+)"
    match = re.search(pattern, text_to_search, re.DOTALL)

    if match:
        return int(match.group(1))
    else:
        raise ValueError(f"No number found after **Final Answer** in: {text_to_search}")



def visualize_smc_sampling_steps(
    xts, x0_preds, rewards=None, save_dir=None,
    num_samples=3, particles_per_sample=2
):
    os.makedirs(save_dir, exist_ok=True)
    num_steps = len(xts)
    total_particles = num_samples * particles_per_sample

    for step in range(num_steps):
        # 如果该步没有 reward，就跳过显示 score
        has_reward = rewards is not None and len(rewards[step]) > 0

        fig, axs = plt.subplots(num_samples, particles_per_sample * 2, figsize=(particles_per_sample * 4.5, num_samples * 2.5))

        if num_samples == 1:
            axs = axs.reshape(1, -1)
        if particles_per_sample == 1:
            axs = axs.reshape(-1, 2)

        for s in range(num_samples):
            for p in range(particles_per_sample):
                idx = s * particles_per_sample + p
                xt = xts[step][idx]
                x0 = x0_preds[step][idx]

                xt_img = TF.to_pil_image(torch.clamp((xt + 1) / 2, 0, 1))
                x0_img = TF.to_pil_image(torch.clamp((x0 + 1) / 2, 0, 1))

                # x_t
                axs[s, p * 2].imshow(xt_img)
                axs[s, p * 2].axis('off')
                axs[s, p * 2].set_title(f"Sample {s} | P{p+1} x_t")

                # x_0_pred + reward
                axs[s, p * 2 + 1].imshow(x0_img)
                axs[s, p * 2 + 1].axis('off')

                if has_reward:
                    score = rewards[step][idx]
                    if score is not None:
                        score_text = f"Score: {score:.2f}"
                    else:
                        score_text = ""
                else:
                    score_text = ""
                axs[s, p * 2 + 1].set_title(f"Sample {s} | P{p+1} x₀\n{score_text}")

        plt.suptitle(f"SMC Sampling - Step {step + 1}", fontsize=14)
        plt.tight_layout()
        save_path = os.path.join(save_dir, f"all_samples_step_{step+1:02d}.png")
        plt.savefig(save_path)
        plt.close(fig)
