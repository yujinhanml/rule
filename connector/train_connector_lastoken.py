import os
import re
from PIL import Image
from torch.utils.data import Dataset, DataLoader, DistributedSampler
import torchvision.transforms as transforms
import torch
import random
import torch
import torch.nn as nn
# --- 环境导入 ---
import sys
import wandb
sys.path.append('/cpfs04/user/hanyujin/rule-gen/rule_tokenizer')
import torch
# from modelling.sit import SiT_models
from modelling.lpips import LPIPS
# from modelling.tokenizer import SoftVQModel
from modelling.samplers import euler_sampler
from utils.model import build_tokenizer
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import ruamel.yaml as yaml
from utils.data import random_crop_arr, center_crop_arr
import torch.nn.functional as F 
from tqdm import trange, tqdm
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor,Qwen2VLForConditionalGeneration
import numpy as np
from qwen_vl_utils import process_vision_info

# Minst-Mnist-JEPAAE

vae_type = 'AE'
model_name = "SiT-B/1"
num_classes =  1 #3
resolution = 64
path_type = "linear"
num_steps = 20
guidance_low = 0.0
guidance_high = 1.0
shuffle_ratio = 1
gpu_id = 2
resolution = 64
num_epochs = 500
recon_weight = 5
ori_weight  = 5
perceptual_loss_weight = 5
epoch_interval = 10
max_token = 25 #25 #37
input_dim = 3584
batch_size = 16
connector_type = 'mlp' # "transformer" # "Covntransformer" #'mlp'
device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

if vae_type == 'JEPA-AE':
    vae_config_path =  "/cpfs04/user/hanyujin/rule-gen/rule_tokenizer/configs/in1k/exp006-aejepadiff-16.yaml"
    vae_ckpt_path = "/cpfs04/user/hanyujin/rule-gen/experiments/tokenizer/mnist-mnist-seq-exp004-aejepadiff-16-jepa1.0-blocktyperandom/checkpoints/0001500.pt"
    # sit_ckpt_path = "/cpfs04/user/hanyujin/rule-gen/experiments/sit/mnist-mnist-seq-000-SiT-B-1-linear-uniform-0001500.pt-jepa1.0-blocktyperandom/checkpoints/0062000.pt"
    vq_mean = 0.24468633210659027 
    vq_std = 1.3076791763305664
elif vae_type == 'AE':
    vae_config_path =  "/cpfs04/user/hanyujin/rule-gen/rule_tokenizer/configs/in1k/exp007-aediff-16.yaml"
    vae_ckpt_path = "/cpfs04/user/hanyujin/rule-gen/experiments/tokenizer/mnist-mnist-seq-exp003-aediff-16/checkpoints/0001500.pt"
    vq_mean = 0.07790039056539536
    vq_std = 0.9384990930557251

local_path = "/cpfs04/shared/CausaLLMs/HuggingfaceModels/Qwen2.5-VL-7B-Instruct"

mllm_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    local_path, torch_dtype="auto", device_map="auto"
)

processor = AutoProcessor.from_pretrained(local_path)

with open(vae_config_path, 'r') as f:
    config_args = yaml.YAML().load(f)
    dataset = config_args['dataset']
    vq_model = config_args['vq_model']

vae, vae_string_name, vae_embed_dim, dit_input_size, _, _, vae_1d = build_tokenizer(vae_config_path, vae_ckpt_path)
vae = vae.to(device).eval()
print(f"Loaded VAE: embed_dim={vae_embed_dim}, input_size={dit_input_size}, 1d={vae_1d}")


transform = transforms.Compose([
    # transforms.Lambda(lambda pil_image: random_crop_arr(pil_image, resolution)),
    transforms.Resize(resolution),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
])

# class DigitSeqDataset(Dataset):
#     def __init__(self, image_dir, transform=None, shuffle_ratio=1.0):
#         self.transform = transform
#         self.samples = []  # [(digits, label_str, image_path)]
#         self.label_to_paths = {}  # label_str → list of image_paths
#         self.shuffle_ratio = shuffle_ratio

#         for fname in sorted(os.listdir(image_dir)):
#             if fname.endswith(".png") and fname.startswith("seq_"):
#                 img_path = os.path.join(image_dir, fname)
#                 match = re.match(r"seq_(\d+)-(\d+)_\d+\.png", fname)
#                 if match:
#                     start = int(match[1])
#                     end = int(match[2])
#                     digits = list(range(start, end + 1))
#                     label_str = ",".join(str(d) for d in digits)

#                     # 记录样本唯一路径
#                     self.samples.append((digits, label_str, img_path))

#                     if label_str not in self.label_to_paths:
#                         self.label_to_paths[label_str] = []
#                     self.label_to_paths[label_str].append(img_path)

#         self.total = len(self.samples)

#     def __len__(self):
#         return self.total

#     def __getitem__(self, idx):
#         digits, label_str, image_path_prompt = self.samples[idx]
#         prompt = f"An image of digits {label_str}"

#         # === image 一定是 prompt 对应图像 ===
#         image = Image.open(image_path_prompt).convert("RGB")
#         if self.transform:
#             image = self.transform(image)

#         # === 是否打乱 image_path ===
#         if random.random() < self.shuffle_ratio:
#             target_index = (idx + 1) % self.total
#             _, shifted_label_str, _ = self.samples[target_index]
#             shifted_candidates = self.label_to_paths.get(shifted_label_str, [])
#             if not shifted_candidates:
#                 raise ValueError(f"No image found for shifted label: {shifted_label_str}")
#             img_path = random.choice(shifted_candidates)
#         else:
#             img_path = image_path_prompt  # 不打乱，保留原图路径

#         return image, prompt, img_path

# data_path = "/cpfs04/user/hanyujin/rule-gen/datasets/mnist-mnist-seq/images"
# dataset = DigitSeqDataset(data_path, transform=transform,shuffle_ratio=shuffle_ratio)



class DigitSeqNegDataset(Dataset):
    def __init__(self, pos_dir, neg_dir, transform=None):
        self.samples = []
        self.transform = transform

        # 把 neg 中的图片放到字典中：编号后缀 → 文件名
        neg_dict = {
            re.search(r"_(\d{4})\.png", fname).group(1): fname
            for fname in os.listdir(neg_dir)
            if fname.endswith(".png")
        }

        for fname in os.listdir(pos_dir):
            if not fname.endswith(".png"):
                continue

            suffix_match = re.search(r"_(\d{4})\.png", fname)
            if not suffix_match:
                continue
            suffix = suffix_match.group(1)

            if suffix not in neg_dict:
                continue  

            pos_digits = list(map(int, re.search(r"mnist_(.*?)_\d{4}\.png", fname).group(1).split("-")))
            neg_fname = neg_dict[suffix]
            neg_digits = list(map(int, re.search(r"mnist_(.*?)_\d{4}\.png", neg_fname).group(1).split("-")))

            for i in range(4):
                if pos_digits[i] != neg_digits[i]:
                    changed_index = i
                    old_digit = neg_digits[i]
                    new_digit = pos_digits[i]
                    break

            region_names = ["top-left", "top-right", "bottom-left", "bottom-right"]
            # prompt = f"You should modify the digit located at the {region_names[changed_index]} position and set it to {new_digit}, while keeping the rest of the image unchanged."
            prompt = f"An image of digits {pos_digits}"
            # prompt_old = f"An image of digits {neg_digits}."
            # prompt = f"The image shows four digits. The digit in the {region_names[changed_index]} corner is incorrect: it shows {old_digit}, but it should be {new_digit}. Please correct it."
            self.samples.append({
                "image_path_pos": os.path.join(pos_dir, fname),
                "image_path_neg": os.path.join(neg_dir, neg_fname),
                "prompt": prompt
            })
            # self.samples.append({
            #     "image_path_pos": os.path.join(pos_dir, fname),
            #     "image_path_neg": os.path.join(neg_dir, neg_fname),
            #     "prompt_new": prompt_new,
            #     "prompt_old": prompt_old
            # })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        entry = self.samples[idx]
        image_pos = Image.open(entry["image_path_pos"]).convert("RGB")
        image_neg = Image.open(entry["image_path_neg"]).convert("RGB")
        if self.transform:
            image_pos = self.transform(image_pos)
            image_neg = self.transform(image_neg)
        # return image_pos, entry["prompt_new"], entry["prompt_old"], entry["image_path_neg"], image_neg
        return image_pos, entry["prompt"], entry["image_path_neg"], image_neg


data_path = "/cpfs04/user/hanyujin/rule-gen/datasets/mnist-mnist-seq-pos/images"
data_path_neg = "/cpfs04/user/hanyujin/rule-gen/datasets/mnist-mnist-seq-neg-1/images"

dataset = DigitSeqNegDataset(
    pos_dir = data_path,
    neg_dir = data_path_neg,
    transform=transform
)

train_loader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
    drop_last=True,
)

def samples_con(con, xT, decoder, vq_mean=vq_mean, vq_std=vq_std):
    con = con * vq_std + vq_mean
    B, C, H, W = con.shape
    con = con.view(B, C, H * W).permute(0, 2, 1)

    sampling_kwargs_img = dict(
        model=decoder,
        latents=xT,
        y=con,
        num_steps=num_steps,
        heun=False,
        cfg_scale=1,
        guidance_low=guidance_low,
        guidance_high=guidance_high,
        path_type=path_type,
        num_classes=num_classes
    )
    return euler_sampler(**sampling_kwargs_img).to(torch.float32)


class MLPProjector(nn.Module):
    def __init__(self, input_tokens=1, input_dim=input_dim, output_tokens=256, output_dim=32, hidden_dim=2048, depth=2): #if full last layer set input_tokens=max_token
        super().__init__()
        self.input_dim = input_dim
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.output_dim = output_dim
        self.flatten_dim = input_tokens * input_dim
        self.output_flat_dim = output_tokens * output_dim

        layers = [nn.Linear(self.flatten_dim, hidden_dim)]
        for _ in range(1, depth):
            layers.append(nn.GELU())
            layers.append(nn.Linear(hidden_dim, hidden_dim))
        layers.append(nn.GELU())
        layers.append(nn.Linear(hidden_dim, self.output_flat_dim))

        self.mlp = nn.Sequential(*layers)

    def forward(self, hidden_states):
        # B, T, D = hidden_states.shape
        # assert T == self.input_tokens and D == self.input_dim, \
        #     f"Expected input shape [B, {self.input_tokens}, {self.input_dim}], got [B, {T}, {D}]"
        B, _ = hidden_states.shape
        x = hidden_states.view(B, -1)  # [B, 37*3584]
        x = self.mlp(x)                # [B, 256*32]
        x = x.view(B, self.output_tokens, self.output_dim)  # [B, 256, 32]
        return x



class TransformerProjector(nn.Module):
    def __init__(
        self,
        input_tokens = max_token,   # T
        input_dim = input_dim,      # D
        output_tokens=256,
        output_dim=32,
        num_layers=4,
        num_heads=8,
        dim_feedforward=2048,
        dropout=0.1,
    ):
        super().__init__()
        self.input_tokens = input_tokens
        self.input_dim = input_dim
        self.output_tokens = output_tokens
        self.output_dim = output_dim

        # 可学习的位置编码
        self.pos_embed = nn.Parameter(torch.randn(1, input_tokens, input_dim))

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 从 Transformer 输出的每个 token 上做 linear 映射 → 输出 token 序列
        self.proj_head = nn.Sequential(
            nn.Linear(input_tokens * input_dim, output_tokens * output_dim)
        )

    def forward(self, hidden_states):  # [B, T, D]
        B, T, D = hidden_states.shape
        assert T == self.input_tokens and D == self.input_dim, \
            f"Expected input shape [B, {self.input_tokens}, {self.input_dim}], got [B, {T}, {D}]"

        x = hidden_states + self.pos_embed  # 加上位置编码
        x = self.transformer(x)  # [B, T, D]

        x = x.view(B, -1)  # flatten [B, T * D]
        x = self.proj_head(x)  # [B, output_tokens * output_dim]
        x = x.view(B, self.output_tokens, self.output_dim)
        return x


class ConvTransformerProjector(nn.Module):
    def __init__(
        self,
        input_tokens = max_token,   # T
        input_dim = input_dim,      # D
        output_tokens=256,
        output_dim=32,
        num_layers=4,
        num_heads=8,
        dim_feedforward=2048,
        dropout=0.1,
    ):
        super().__init__()
        self.input_tokens = input_tokens
        self.input_dim = input_dim
        self.output_tokens = output_tokens
        self.output_dim = output_dim

        # Learnable pos encoding
        self.pos_embed = nn.Parameter(torch.randn(1, input_tokens, input_dim))

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Conv1D across token dim (optional enhancement)
        self.conv1d_token = nn.Sequential(
            nn.Conv1d(input_tokens, output_tokens, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(output_tokens, output_tokens, kernel_size=3, padding=1),
        )

        # Projection to final output
        self.proj_head = nn.Linear(input_dim, output_dim)

    def forward(self, hidden_states):  # [B, T, D]
        B, T, D = hidden_states.shape
        assert T == self.input_tokens and D == self.input_dim, \
            f"Expected input shape [B, {self.input_tokens}, {self.input_dim}], got [B, {T}, {D}]"

        x = hidden_states + self.pos_embed                     # [B, T, D]
        x = self.transformer(x)                                # [B, T, D]
        x = self.proj_head(x)                                  # [B, T, output_dim]

        # Conv1D over token dimension → [B, T, output_dim] → [B, output_tokens, output_dim]
        x = self.conv1d_token(x)                               # [B, output_tokens, output_dim]

        return x  # shape: [B, 256, 32]



if connector_type == "mlp":
    connector = MLPProjector().to(device)
elif connector_type == "transformer":
    connector = TransformerProjector().to(device)
elif connector_type == "Covntransformer":
    connector = ConvTransformerProjector().to(device)
print(f"The connector is {connector_type}")
# 冻结 mllm_model 和 vae 的参数
for param in mllm_model.parameters():
    param.requires_grad = False

for param in vae.parameters():
    param.requires_grad = False

# Step 2: dummy 输入初始化 projector 参数（以当前 batch 的 token 数为准）
# with torch.no_grad():
#     dummy_hidden = torch.zeros(1, max_token, input_dim).to(device)  # or replace 366 with actual token count
#     _ = connector(dummy_hidden)


optimizer = torch.optim.Adam(connector.parameters(), lr=1e-4)
wandb_logger = wandb.init(
    project="tokenizer",
    name=f"Edit-{vae_type}-connector{connector_type}-mnist-mnist-epoch{num_epochs}-recon_weight{recon_weight}-ori_weight{ori_weight}-perceptual{perceptual_loss_weight}-shuffle{shuffle_ratio}-lastoken"
)

real_buffer = []
fake_buffer = []
truth_buffer = []
fail_buffer = []
global_step = 0

if perceptual_loss_weight > 0:
    lpips_loss = LPIPS().eval().to(device)

for epoch in trange(num_epochs):
    for img_tensor, text_prompt, img_path, img_neg in tqdm(train_loader, desc=f"Epoch {epoch}"):
        B = len(text_prompt)
        xT = torch.randn(B, 3, resolution, resolution, device=device)

        # === Step 1: encode ground-truth latent ===
        img_tensor = img_tensor.to(device)  # [B, 3, H, W]
        img_neg = img_neg.to(device)
        with torch.no_grad():
            quant, _, _ = vae.encode(img_tensor)  # [B, 256, 32]

        # === Step 2: 构造 Qwen2.5-VL 输入 ===
        messages_list = [
            [{
                "role": "user",
                "content": [
                    {"type": "image", "image": img_path[i]},
                    {"type": "text", "text": text_prompt[i]}
                ]
            }] for i in range(B)
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
            return_tensors="pt"
            # padding="max_length",
            # truncation=True,
            # max_length=max_token,
        ).to(device)

        with torch.no_grad():
            outputs = mllm_model(**inputs, output_hidden_states=True, return_dict=True)
        hidden_states = outputs.hidden_states[-1][:, -1, :].to(torch.float32)  # [B, max_token, 3584]
        # print("hidden_states:",hidden_states.shape)
        
        # === Step 3: project to latent space ===
        project_latents = connector(hidden_states)  # [B, 256, 32]

        # === Step 4: compute loss ===
        B, N, C = quant.shape
        H = W = int(N ** 0.5)
        quant_input = quant.permute(0, 2, 1).view(B, C, H, W)  # [B, 32, 16, 16]
        project_latents_input = project_latents.permute(0, 2, 1).view(B, C, H, W)

        semantic_loss = F.mse_loss(project_latents, quant.detach())

        samples_real = samples_con(quant_input, xT, vae.decoder)
        samples_fake = samples_con(project_latents_input, xT, vae.decoder)

        recon_pixel_loss = F.mse_loss(samples_fake, samples_real)
        ori_pixel_loss = F.mse_loss(samples_fake, img_tensor)

        if perceptual_loss_weight > 0:
            ori_perceptual_loss = lpips_loss(img_tensor, samples_fake).mean()
        else:
            ori_perceptual_loss = torch.tensor(0., device=img_tensor.device, dtype=img_tensor.dtype)


        total_loss = semantic_loss + recon_weight * recon_pixel_loss + ori_weight * ori_pixel_loss + perceptual_loss_weight * ori_perceptual_loss

        wandb_logger.log({
            "semantic_loss": semantic_loss.item(),
            "recon_pixel_loss": recon_pixel_loss.item(),
            "ori_pixel_loss": ori_pixel_loss.item(),
            "ori_perceptual_loss": ori_perceptual_loss.item(),
            "total_loss": total_loss.item(),
        }, step=global_step)

        # === Step 5: backward ===
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        global_step += 1

        # === 可视化前8张样本 ===
        if len(real_buffer) <= 8:
            for i in range(min(B, 8)):
                real = ((torch.clamp(samples_real[i], -1.0, 1.0))+1)/2
                fake = ((torch.clamp(samples_fake[i], -1.0, 1.0))+1)/2
                truth = ((torch.clamp(img_tensor[i], -1.0, 1.0))+1)/2
                fail = ((torch.clamp(img_neg[i], -1.0, 1.0))+1)/2
                real_buffer.append(real)
                fake_buffer.append(fake)
                truth_buffer.append(truth)
                fail_buffer.append(fail)

    # === 拼图可视化 ===
    row_pairs = [
    torch.cat([t, r, fa, f], dim=2)
    for t, r, fa, f in zip(truth_buffer[:8], real_buffer[:8], fail_buffer[:8], fake_buffer[:8])
    ]

    grid = torch.cat(row_pairs, dim=1)             # [3, 8H, 3W]
    grid_np = grid.permute(1, 2, 0).cpu().numpy()  # [8H, 3W, 3] → for wandb.Image

    wandb_logger.log({
        "comparison_grid": wandb.Image(grid_np, caption=f"Epoch {epoch} real(left) vs fake(right)")
    })
    real_buffer.clear()
    fake_buffer.clear()
    truth_buffer.clear()
    fail_buffer.clear()
    # === 保存 checkpoint ===
    if epoch % epoch_interval == 0:
        print(f"[Epoch {epoch}] semantic_loss: {semantic_loss.item():.4f}, recon_pixel_loss: {recon_pixel_loss.item():.4f}, ori_pixel_loss: {ori_pixel_loss.item():.4f}, total_loss: {total_loss.item():.4f}")

        ckpt_dir = f"/cpfs04/user/hanyujin/rule-gen/rule_tokenizer/connector/checkpoints/Edit-{vae_type}-connector{connector_type}_weight{recon_weight}-ori_weight{ori_weight}-perceptual{perceptual_loss_weight}_shuffle{shuffle_ratio}-lastoken"
        os.makedirs(ckpt_dir, exist_ok=True)
        ckpt_path = os.path.join(ckpt_dir, f"epoch{epoch}.pt")
        torch.save(connector.state_dict(), ckpt_path)
        print(f"✅ Saved MLP checkpoint to {ckpt_path}")


# for epoch in trange(num_epochs):
#     for img_tensor, text_prompt, text_prompt_neg, img_path, img_neg in tqdm(train_loader, desc=f"Epoch {epoch}"):
#         B = len(text_prompt)
#         xT = torch.randn(B, 3, resolution, resolution, device=device)

#         # === Step 1: encode ground-truth latent ===
#         img_tensor = img_tensor.to(device)  # [B, 3, H, W]
#         img_neg = img_neg.to(device)
#         with torch.no_grad():
#             quant, _, _ = vae.encode(img_tensor)  # [B, 256, 32]
#             quant_neg, _, _ = vae.encode(img_neg)

#         # === Step 2: 构造 Qwen2.5-VL 输入 ===
#         messages_list = [
#             [{
#                 "role": "user",
#                 "content": [
#                     {"type": "image", "image": img_path[i]},
#                     {"type": "text", "text": text_prompt[i]}
#                 ]
#             }] for i in range(B)
#         ]
#         # print("text_prompt:",text_prompt)
#         texts = [
#             processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
#             for msg in messages_list
#         ]
#         image_inputs, video_inputs = process_vision_info([m[0] for m in messages_list])

#         inputs = processor(
#             text=texts,
#             images=image_inputs,
#             videos=video_inputs,
#             return_tensors="pt",
#             padding=True,
#             truncation=True,
#             max_length=max_token,
#         ).to(device)
#         # print("inputs:",inputs['input_ids'].shape)

#         with torch.no_grad():
#             outputs = mllm_model(**inputs, output_hidden_states=True, return_dict=True)
#         # hidden_states = outputs.hidden_states[-1].to(torch.float32)  # [B, max_token, 3584]

#         # Base prompt
#         base_messages_list = [
#             [{
#                 "role": "user",
#                 "content": [
#                     {"type": "image", "image": img_path[i]},
#                     {"type": "text", "text": text_prompt_neg[i]}
#                 ]
#             }] for i in range(B)
#         ]

#         base_texts = [
#             processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
#             for msg in base_messages_list
#         ]

#         base_inputs = processor(
#             text=base_texts,
#             images=image_inputs,
#             videos=video_inputs,
#             return_tensors="pt",
#             padding=True,
#             truncation=True,
#             max_length=max_token,
#         ).to(device)

#         # === 获取最后一层 hidden states 并做差异提取 ===
#         with torch.no_grad():
#             outputs = mllm_model(**inputs, output_hidden_states=True, return_dict=True)
#             base_outputs = mllm_model(**base_inputs, output_hidden_states=True, return_dict=True)

#         # hidden_states = (outputs.hidden_states[-1][:, -1, :] - base_outputs.hidden_states[-1][:, -1, :]).to(torch.float32)  # [B, max_token, D]
#         hidden_states = (outputs.hidden_states[-1][:, -1, :]).to(torch.float32)