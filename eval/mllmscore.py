from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor,Qwen2VLForConditionalGeneration
from qwen_vl_utils import process_vision_info
model_name = "QVQ-72B-Preview" # "Qwen2.5-VL-72B-Instruct"
local_path = f"/cpfs04/shared/CausaLLMs/HuggingfaceModels/{model_name}"
image_size = 64
if model_name == "Qwen2.5-VL-72B-Instruct" or model_name == "Qwen2.5-VL-7B-Instruct":
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            local_path, torch_dtype="auto", device_map="auto"
        )
if model_name == "QVQ-72B-Preview":
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            local_path, torch_dtype="auto", device_map="auto"
        )

processor = AutoProcessor.from_pretrained(local_path)


import re
def evaluate_image(image_path,image_size) -> str:
    """
    Evaluates the given image for internal visual consistency and returns a score.
    
    The evaluation is performed by prompting the model with a message asking to rate the image
    from 0 (inconsistent) to 10 (completely harmonious and consistent), following the format:
    "The score is {your_score}. The reason is {your_reason}."
    
    Args:
        image_path (str): The path to the image.
    
    Returns:
        str: The generated output text containing the score and the reason.
    """
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path,"resized_height":image_size,"resized_width":image_size},
                {"type": "text", "text": (
                    "You are an expert in visual anomaly detection, focusing primarily on the shape and outline of objects. You will see an image of an object in front of a mirror and its reflection. Specifically, please pay close attention to the outline and shape differences between the object in front of the mirror and its reflection, and consider whether they could reasonably be the front and back of the same object. If you think any inconsistency stems solely from color or texture design differences between the object and its reflection, please disregard it, since the front and back (or reflection) of an object may naturally vary in these aspects. Also keep in mind that perspective can cause some shape distortion in the reflection, but the overall outline should remain consistent."
                    "Please rate each image individually on a scale from 0 (inconsistent) to 10 (completely harmonious and consistent). The specific scoring criteria are as follows:"
                    "Score 10: Perfect match in shape and outline, considering the mirror flip."  
                    "Score 9: Minor distortions due to perspective, but overall shape is recognizable."  
                    "Score 8: Slight inconsistencies in shape, possibly due to object positioning or minor defects."  
                    "Score 7: Moderate inconsistencies, where the general shape is similar but there are noticeable differences."  
                    "Score 6: Significant inconsistencies, making it hard to recognize the reflection as the object’s reflection."  
                    "Score 5: Equal parts consistent and inconsistent, borderline case."  
                    "Score 4: Reflection shows a completely different shape, but some elements are similar."  
                    "Score 3: Reflection bears little resemblance to the object."  
                    "Score 2: Only a few parts of the reflection match the object."  
                    "Score 1: Almost no similarity in shape and outline."  
                    "Score 0: Reflection is completely different, no similarity at all."
                    "Please think this step by step and answer in the the format: The score is {your_score}. The reason is {your_reason}."
                )},
            ],
        }
    ]

    # Preparation for inference
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")

    # Inference: Generation of the output
    generated_ids = model.generate(**inputs, max_new_tokens=9000)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    

    if model_name == "QVQ-72B-Preview":
        print("image_path:",image_path,"output_text:",output_text)
        return extract_score_after_final_answer(output_text) 
    if model_name == "Qwen2.5-VL-72B-Instruct" or model_name == "Qwen2.5-VL-7B-Instruct":
        pattern = r"The score is (\d+)"
        match = re.search(pattern, output_text[0])
        if match:
            return int(match.group(1))
        else:
            raise ValueError(f"Score not found in output: {output_text[0]}")


def extract_score_after_final_answer(output_text):
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

# def evaluate_image(image_path: str) -> str:
#     """
#     Evaluates the given image for internal visual consistency and returns a score.
    
#     The evaluation is performed by prompting the model with a message asking to rate the image
#     from 0 (inconsistent) to 10 (completely harmonious and consistent), following the format:
#     "The score is {your_score}. The reason is {your_reason}."
    
#     Args:
#         image_path (str): The path to the image.
    
#     Returns:
#         str: The generated output text containing the score and the reason.
#     """
#     messages = [
#         {
#             "role": "user",
#             "content": [
#                 {"type": "image", "image": image_path,"resized_height":image_size,"resized_width":image_size},
#                 {"type": "text", "text": (
#                     "You are an expert in visual anomaly detection, responsible for examining the internal consistency of each image. You will see images of an object and its reflection in a mirror." 
#                     "Specifically, please pay close attention to the outline and shape differences between the object in front of the mirror and its reflection, and consider whether they could reasonably be the front and back of the same object. If you notice any inconsistencies that are purely due to color, texture, or pattern differences, please ignore them, since the front and back (or reflection) of an object can naturally vary in these aspects. Instead, pay special attention to whether the overall shape and outline of the object match its reflection, accounting for normal perspective distortion. If the outlines or proportions differ significantly, consider the reflection inconsistent."
#                     "Please rate each image individually on a scale from 0 (inconsistent) to 10 (completelyconsistent)."
#                     "Please follow the format: The score is {your_score}. "
#                     "The reason is {your_reason}."
#                 )},
#             ],
#         }
#     ]

#     # Preparation for inference
#     text = processor.apply_chat_template(
#         messages, tokenize=False, add_generation_prompt=True
#     )
#     image_inputs, video_inputs = process_vision_info(messages)
#     inputs = processor(
#         text=[text],
#         images=image_inputs,
#         videos=video_inputs,
#         padding=True,
#         return_tensors="pt",
#     )
#     inputs = inputs.to("cuda")

#     # Inference: Generation of the output
#     generated_ids = model.generate(**inputs, max_new_tokens=128)
#     generated_ids_trimmed = [
#         out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
#     ]
#     output_text = processor.batch_decode(
#         generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
#     )
    
#     # 返回生成的评分信息（例如字符串形式）

#     print("image_path:",image_path,"output_text:",output_text)
#     pattern = r"The score is (\d+)"
#     match = re.search(pattern, output_text[0])
#     if match:
#         return int(match.group(1))
#     else:
#         raise ValueError(f"Score not found in output: {output_text[0]}")

# def evaluate_image(image_path: str) -> str:
#     """
#     Evaluates the given image for internal visual consistency and returns a score.
    
#     The evaluation is performed by prompting the model with a message asking to rate the image
#     from 0 (inconsistent) to 10 (completely harmonious and consistent), following the format:
#     "The score is {your_score}. The reason is {your_reason}."
    
#     Args:
#         image_path (str): The path to the image.
    
#     Returns:
#         str: The generated output text containing the score and the reason.
#     """
#     messages = [
#         {
#             "role": "user",
#             "content": [
#                 {"type": "image", "image": image_path},
#                 {"type": "text", "text": (
#                  "You are an expert in visual anomaly detection, responsible for examining the internal consistency of each image. You will see images of an object and its reflection in a mirror." 
#                     "For example, to what extent does the object reflected in the mirror match the object in front of it? "
#                     "Please rate each image individually on a scale from 0 (inconsistent) to 10 (completely "
#                     "harmonious and consistent). Please follow the format: The score is {your_score}. "
#                     "The reason is {your_reason}."
#                 )}
#             ],
#         }
#     ]
#                     # "You are an expert in visual anomaly detection, responsible for examining "
#                     # "the internal consistency of each image. For example, do the relationships between "
#                     # "objects in the image follow reasonable rules? For example, do the objects’ shadows, "
#                     # "specular reflections, and water surface refractions adhere to natural laws? "
#                     # "Please rate each image individually on a scale from 0 (inconsistent) to 10 (completely "
#                     # "harmonious and consistent). Please follow the format: The score is {your_score}. "
#                     # "The reason is {your_reason}."
#     # Preparation for inference
#     text = processor.apply_chat_template(
#         messages, tokenize=False, add_generation_prompt=True
#     )
#     image_inputs, video_inputs = process_vision_info(messages)
#     inputs = processor(
#         text=[text],
#         images=image_inputs,
#         videos=video_inputs,
#         padding=True,
#         return_tensors="pt",
#     )
#     inputs = inputs.to("cuda")

#     # Inference: Generation of the output
#     generated_ids = model.generate(**inputs, max_new_tokens=128)
#     generated_ids_trimmed = [
#         out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
#     ]
#     output_text = processor.batch_decode(
#         generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
#     )
    
#     # 返回生成的评分信息（例如字符串形式）
#     pattern = r"The score is (\d+)"
#     match = re.search(pattern, output_text[0])
#     if match:
#         return int(match.group(1))
#     else:
#         raise ValueError(f"Score not found in output: {output_text[0]}")

import os
import random
import numpy as np
from PIL import Image
import torch
from tqdm import tqdm
import tempfile

# 图像文件夹路径
image_dir = "/cpfs04/user/hanyujin/rule-gen/datasets/mirrors/left"
image_dir_1 = "/cpfs04/user/hanyujin/rule-gen/experiments/samples/AE-Diff-16-mirror-SiT-B-1-linear-vae0040000.pt-size-64cfg-4.0-seed-0"
image_dir_2 = "/cpfs04/user/hanyujin/rule-gen/experiments/samples/AE-JEPA-Diff-16-mirror-SiT-B-1-linear-vae0040000.pt-size-64cfg-4.0-seed-0"
# 获取所有 PNG 文件
image_files = [f for f in os.listdir(image_dir) if f.endswith(".png")]
image_files_1 = [f for f in os.listdir(image_dir_1) if f.endswith(".png")]
image_files_2 = [f for f in os.listdir(image_dir_2) if f.endswith(".png")]
# 随机采样100张图片（如果图片总数不足，则全部使用）
sample_size = min(10, len(image_files))
sampled_files = random.sample(image_files, sample_size) #sorted(image_files)[:sample_size] #random.sample(image_files_1, sample_size)

sample_size = min(10, len(image_files_1))
sampled_files_1 = random.sample(image_files_1, sample_size) #sorted(image_files_1)[:sample_size] #random.sample(image_files_1, sample_size)

sample_size = min(10, len(image_files_2))
sampled_files_2 = random.sample(image_files_2, sample_size) #sorted(image_files_2)[:sample_size] #random.sample(image_files_2, sample_size)

# 用于存储 Qwen 得分
scores_original = []
running_sum = 0
valid_count = 0
pbar = tqdm(sampled_files, desc="Processing Raw Images")
for image_name in pbar:
    image_path = os.path.join(image_dir, image_name)
    try:
        score = evaluate_image(image_path,image_size)
    except Exception as e:
        print(f"Error evaluating original image {image_name}: {e}")
        score = None

    if score is not None:
        running_sum += score
        valid_count += 1

    scores_original.append(score)
    print("Train image scores:", scores_original)
    avg_score = running_sum / valid_count if valid_count > 0 else 0
    pbar.set_postfix(avg_score=f"{avg_score:.4f}")
print("Train image scores:", scores_original)


# 用于存储 Qwen 得分
scores_original_1 = []
running_sum = 0
valid_count = 0
pbar = tqdm(sampled_files_1, desc="Processing AE Images")
for image_name in pbar:
    image_path = os.path.join(image_dir_1, image_name)
    try:
        score = evaluate_image(image_path,image_size)
    except Exception as e:
        print(f"Error evaluating original image {image_name}: {e}")
        score = None

    if score is not None:
        running_sum += score
        valid_count += 1

    scores_original_1.append(score)
    print("AE image scores:", scores_original_1)
    avg_score = running_sum / valid_count if valid_count > 0 else 0
    pbar.set_postfix(avg_score=f"{avg_score:.4f}")
print("AE image scores:", scores_original_1)


scores_original_2 = []
running_sum = 0
valid_count = 0
pbar = tqdm(sampled_files_2, desc="Processing AE-JEPA Images")
for image_name in pbar:
    image_path = os.path.join(image_dir_2, image_name)
    try:
        score = evaluate_image(image_path,image_size)
    except Exception as e:
        print(f"Error evaluating original image {image_name}: {e}")
        score = None

    if score is not None:
        running_sum += score
        valid_count += 1

    scores_original_2.append(score)
    print("JEPA image scores:", scores_original_2)
    avg_score = running_sum / valid_count if valid_count > 0 else 0
    pbar.set_postfix(avg_score=f"{avg_score:.4f}")

print("JEPA image scores:", scores_original_2)


save_dir = "/cpfs04/user/hanyujin/rule-gen/rule_tokenizer/eval/mllmscore/"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
file_path = os.path.join(save_dir, f"scores_{model_name}_{image_size}_ae-jepaae.txt")

with open(file_path, "w") as f:
    f.write("Train image scores: {}\n".format(scores_original))
    f.write("AE image scores: {}\n".format(scores_original_1))
    f.write("JEPA-AE image scores: {}\n".format(scores_original_2))


print(f"Scores saved to {file_path}")

import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from scipy.stats import ttest_ind

data = {
    f"{model_name}": ["RAW"] * len(scores_original)+ ["AE"] * len(scores_original_1) + ["JEPA-AE"] * len(scores_original_2),
    "MLLM Score":scores_original +  scores_original_1 + scores_original_2
}
df = pd.DataFrame(data)

# 计算 t 检验的 p-value
t_stat, p_val = ttest_ind(scores_original_1, scores_original_2, equal_var=False)
p_value_text = f"p = {p_val:.3f}"

# 设置绘图参数
plt.rcParams["axes.labelsize"] = 15
palette = ['#19980624','#0073C2FF', '#EFC000FF']

fig, ax = plt.subplots(figsize=(5, 5), dpi=100, facecolor="w")
ax = sns.barplot(
    x=f"{model_name}", y="MLLM Score", data=df, palette=palette,
    estimator=np.mean, ci="sd", capsize=.1, errwidth=1, errcolor="k",
    ax=ax, edgecolor="k", linewidth=1
)

# 添加 p-value 标注
x1, x2 = 1, 2  # 两个类别在 x 轴上的位置
# 确定标注线的高度：取两个组均值加标准差中的最大值，再加上一些偏移
y = max(np.mean(scores_original_1) + np.std(scores_original_1), np.mean(scores_original_2) + np.std(scores_original_2)) + 1
h = 0.2  # 横线的高度偏移
ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1, c="k")
ax.text((x1+x2)*.5, y+h, "T-test: " + p_value_text, ha='center', va='bottom', color="k")

# 调整坐标轴和网格
ax.tick_params(which='major', direction='in', length=3, width=1., labelsize=14, bottom=False)
for spine in ["top", "left", "right"]:
    ax.spines[spine].set_visible(False)
ax.spines['bottom'].set_linewidth(2)
ax.grid(axis='y', ls='--', c='gray')
ax.set_axisbelow(True)
plt.savefig(f"/cpfs04/user/hanyujin/rule-gen/rule_tokenizer/eval/mllmscore/mmlmscore_{model_name}_{image_size}_ae-jepaae.png")
plt.show()




# from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
# from qwen_vl_utils import process_vision_info
# model_name = "Qwen2.5-VL-72B-Instruct"
# local_path = f"/cpfs04/shared/CausaLLMs/HuggingfaceModels/{model_name}"

# model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
#     local_path, torch_dtype="auto", device_map="auto"
# )

# processor = AutoProcessor.from_pretrained(local_path)

# import re
# def evaluate_image(image_path: str) -> str:
#     """
#     Evaluates the given image for internal visual consistency and returns a score.
    
#     The evaluation is performed by prompting the model with a message asking to rate the image
#     from 0 (inconsistent) to 10 (completely harmonious and consistent), following the format:
#     "The score is {your_score}. The reason is {your_reason}."
    
#     Args:
#         image_path (str): The path to the image.
    
#     Returns:
#         str: The generated output text containing the score and the reason.
#     """
#     messages = [
#         {
#             "role": "user",
#             "content": [
#                 {"type": "image", "image": image_path},
#                 {"type": "text", "text": (
#                     "You are an expert in visual anomaly detection, responsible for examining "
#                     "the internal consistency of each image. For example, do the relationships between "
#                     "objects in the image follow reasonable rules? For example, do the objects’ shadows, "
#                     "specular reflections, and water surface refractions adhere to natural laws? "
#                     "Please rate each image individually on a scale from 0 (inconsistent) to 10 (completely "
#                     "harmonious and consistent). Please follow the format: The score is {your_score}. "
#                     "The reason is {your_reason}."
#                 )}
#             ],
#         }
#     ]

#     # Preparation for inference
#     text = processor.apply_chat_template(
#         messages, tokenize=False, add_generation_prompt=True
#     )
#     image_inputs, video_inputs = process_vision_info(messages)
#     inputs = processor(
#         text=[text],
#         images=image_inputs,
#         videos=video_inputs,
#         padding=True,
#         return_tensors="pt",
#     )
#     inputs = inputs.to("cuda")

#     # Inference: Generation of the output
#     generated_ids = model.generate(**inputs, max_new_tokens=128)
#     generated_ids_trimmed = [
#         out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
#     ]
#     output_text = processor.batch_decode(
#         generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
#     )
    
#     # 返回生成的评分信息（例如字符串形式）
#     pattern = r"The score is (\d+)"
#     match = re.search(pattern, output_text[0])
#     if match:
#         return int(match.group(1))
#     else:
#         raise ValueError(f"Score not found in output: {output_text[0]}")

# import os
# import random
# import numpy as np
# from PIL import Image
# import torch
# from tqdm import tqdm
# import tempfile

# # 图像文件夹路径
# image_dir = "/cpfs04/user/hanyujin/rule-gen/datasets/mirrors/left"

# # 获取所有 PNG 文件
# image_files = [f for f in os.listdir(image_dir) if f.endswith(".png")]

# # 随机采样100张图片（如果图片总数不足，则全部使用）
# sample_size = min(1000, len(image_files))
# sampled_files = random.sample(image_files, sample_size)

# # 用于存储 Qwen 得分
# scores_original = []
# scores_remove_reflection = []
# scores_rotation = []

# for image_name in tqdm(sampled_files, desc="Processing Images"):
#     image_path = os.path.join(image_dir, image_name)

#     # 读取原始图像（image1）
#     image1 = Image.open(image_path).convert("RGB")
#     image1_np = np.array(image1)

#     # 生成去反射图（image2）
#     h, w, c = image1_np.shape
#     upper_half = image1_np[:int(h//3)].copy()  # 取上半部分（这里以 h//3 作为示例区域）
#     x1, x2 = w // 3, 2 * w // 3  # 插值范围
#     for x in range(x1, x2):
#         alpha = (x - x1) / (x2 - x1)
#         # 线性插值：左右两侧的像素均值
#         upper_half[:, x] = (1 - alpha) * upper_half[:, x1 - 1] + alpha * upper_half[:, x2]
#     image2_np = image1_np.copy()
#     image2_np[:int(h//3)] = upper_half
#     image2 = Image.fromarray(image2_np.astype(np.uint8))

#     # 生成旋转图（image3）
#     # image3 = image1.rotate(180)

#     # 对于 image1，直接使用原始路径进行评分
#     try:
#         score1 = evaluate_image(image_path)
#     except Exception as e:
#         print(f"Error evaluating original image {image_name}: {e}")
#         score1 = None

#     # 对于 image2 和 image3，由于 evaluate_image 需要图片路径，因此暂存到临时文件中
#     # score2, score3 = None, None
#     # try:
#     #     with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
#     #         tmp_path2 = tmp_file.name
#     #         image2.save(tmp_path2)
#     #     score2 = evaluate_image(tmp_path2)
#     # except Exception as e:
#     #     print(f"Error evaluating reflection-removed image {image_name}: {e}")
#     # finally:
#     #     if os.path.exists(tmp_path2):
#     #         os.remove(tmp_path2)
    
#     # try:
#     #     with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
#     #         tmp_path3 = tmp_file.name
#     #         image3.save(tmp_path3)
#     #     score3 = evaluate_image(tmp_path3)
#     # except Exception as e:
#     #     print(f"Error evaluating rotated image {image_name}: {e}")
#     # finally:
#     #     if os.path.exists(tmp_path3):
#     #         os.remove(tmp_path3)
    
#     scores_original.append(score1)
#     # scores_remove_reflection.append(score2)
#     # scores_rotation.append(score3)

# # 打印或进一步处理得到的得分
# print("Original image scores:", scores_original)
# # print("Reflection removed image scores:", scores_remove_reflection)
# # print("Rotated image scores:", scores_rotation)

# save_dir = "/cpfs04/user/hanyujin/rule-gen/rule_tokenizer/eval/mllmscore/"
# if not os.path.exists(save_dir):
#     os.makedirs(save_dir)

# # 指定保存的文件名
# file_path = os.path.join(save_dir, f"scores_{model_name}.txt")

# with open(file_path, "w") as f:
#     f.write("Original image scores: {}\n".format(scores_original))
#     f.write("Reflection removed image scores: {}\n".format(scores_remove_reflection))
#     # f.write("Rotated image scores: {}\n".format(scores_rotation))

# print(f"Scores saved to {file_path}")

# import matplotlib.pyplot as plt
# import matplotlib.pyplot as plt
# import numpy as np
# import seaborn as sns
# import pandas as pd
# from scipy.stats import ttest_ind

# data = {
#     "Qwen2.5-VL-7B": ["Raw"] * len(scores_original) + ["Pertub"] * len(scores_remove_reflection),
#     "MLLM Score": scores_original + scores_remove_reflection
# }
# df = pd.DataFrame(data)

# # 计算 t 检验的 p-value
# t_stat, p_val = ttest_ind(scores_original, scores_remove_reflection, equal_var=False)
# p_value_text = f"p = {p_val:.3f}"

# # 设置绘图参数
# plt.rcParams["axes.labelsize"] = 15
# palette = ['#0073C2FF', '#EFC000FF']

# fig, ax = plt.subplots(figsize=(5, 5), dpi=100, facecolor="w")
# ax = sns.barplot(
#     x="Qwen2.5-VL-7B", y="MLLM Score", data=df, palette=palette,
#     estimator=np.mean, ci="sd", capsize=.1, errwidth=1, errcolor="k",
#     ax=ax, edgecolor="k", linewidth=1
# )

# # 添加 p-value 标注
# x1, x2 = 0, 1  # 两个类别在 x 轴上的位置
# # 确定标注线的高度：取两个组均值加标准差中的最大值，再加上一些偏移
# y = max(np.mean(scores_original) + np.std(scores_original), np.mean(scores_remove_reflection) + np.std(scores_remove_reflection)) + 1
# h = 0.2  # 横线的高度偏移
# ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1, c="k")
# ax.text((x1+x2)*.5, y+h, "T-test: " + p_value_text, ha='center', va='bottom', color="k")

# # 调整坐标轴和网格
# ax.tick_params(which='major', direction='in', length=3, width=1., labelsize=14, bottom=False)
# for spine in ["top", "left", "right"]:
#     ax.spines[spine].set_visible(False)
# ax.spines['bottom'].set_linewidth(2)
# ax.grid(axis='y', ls='--', c='gray')
# ax.set_axisbelow(True)
# plt.savefig(f"/cpfs04/user/hanyujin/rule-gen/rule_tokenizer/eval/mllmscore/mmlmscore_{model_name}.png")
# plt.show()

