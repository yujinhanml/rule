import os
import torch
import numpy as np
from tqdm import tqdm
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import sys
sys.path.append('/cpfs04/user/hanyujin/rule-gen/rule_tokenizer')
# from torchvision import transforms
# from utils.data import random_crop_arr, center_crop_arr
import ruamel.yaml as yaml
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
sys.path.append('/cpfs04/user/hanyujin/rule-gen/rule_tokenizer')
from utils.model import build_tokenizer
from torchvision.datasets import DatasetFolder
vae_config = "/cpfs04/user/hanyujin/rule-gen/rule_tokenizer/configs/in1k/exp006-aejepadiff-16.yaml"
# "/cpfs04/user/hanyujin/rule-gen/rule_tokenizer/configs/in1k/exp007-aediff-16.yaml"
vae_ckpt = "/cpfs04/user/hanyujin/rule-gen/experiments/tokenizer/mirror-exp000-aejepadiff-16-jepa1.0-blocktyperandom/checkpoints/0040000.pt"
# "/cpfs04/user/hanyujin/rule-gen/experiments/tokenizer/mirror-exp001-aediff-16/checkpoints/0040000.pt"
vae, vae_model, vae_embed_dim, dit_input_size, latents_bias, latents_scale, vae_1d = build_tokenizer(vae_config, vae_ckpt)
if vae_config is not None:
    with open(vae_config, 'r', encoding='utf-8') as f:
        file_yaml = yaml.YAML()
        config_args = file_yaml.load(f)
# 数据路径
data_path = "/cpfs04/user/hanyujin/rule-gen/datasets/mirrors_train"
# transform = transforms.Compose([
#     transforms.Lambda(lambda pil_image: random_crop_arr(pil_image, 64)),
#     transforms.RandomHorizontalFlip(),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
# ])
transform = Compose([
Resize((64, 64)),
ToTensor(),
Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)])
# 创建 Dataset 和 DataLoader
dataset = ImageFolder(data_path, transform=transform)
# dataset = DatasetFolder(data_path, loader=lambda x: Image.open(x).convert("RGB"), extensions=("png", "jpg"), transform=transform)
dataloader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=4)  # 不打乱，保证一致

# 确保 VAE 处于评估模式
vae.cuda().eval()

# 存储所有 latent features
all_latents = []

# 遍历整个数据集
for imgs, _ in tqdm(dataloader, desc="Extracting Latents"):
    imgs = imgs.cuda()  # 只对图片进行 CUDA 迁移
    with torch.no_grad():
        latent_sample_ext, _, _ = vae.encode(imgs)  # 编码
        all_latents.append(latent_sample_ext.cpu().numpy())  # 存入 list（先放到 CPU）

# 拼接所有 latent
all_latents = np.concatenate(all_latents, axis=0)  # [N, C, H, W] 或其他 shape

# **保存 latent 到 npz**
save_path = f"/cpfs04/user/hanyujin/rule-gen/rule_tokenizer/eval/latents/{config_args['dataset']}-{config_args['vq_model']}.npz"
os.makedirs(os.path.dirname(save_path), exist_ok=True)
np.savez(save_path, latents=all_latents)

print(f"Saved all latents to {save_path}, shape: {all_latents.shape}")