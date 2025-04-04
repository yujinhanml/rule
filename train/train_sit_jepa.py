import argparse
import copy
from copy import deepcopy
import logging
import os
from pathlib import Path
from collections import OrderedDict
import json
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from glob import glob 
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
import ruamel.yaml as yaml
from modelling.sit import SiT_models
from losses.sit_loss import SILoss

import wandb
import math
from torchvision.utils import make_grid
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torchvision.transforms import Normalize
from torchvision.datasets import ImageFolder
from torchvision import transforms

from utils.data import CachedFolder
from utils.model import build_tokenizer

logger = get_logger(__name__)



def array2grid(x):
    nrow = round(math.sqrt(x.size(0)))
    x = make_grid(x.clamp(0, 1), nrow=nrow, value_range=(0, 1))
    x = x.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    return x


@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        name = name.replace("module.", "")
        name = name.replace("_orig_mod.", "")
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    logging.basicConfig(
        level=logging.INFO,
        format='[\033[34m%(asctime)s\033[0m] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
    )
    logger = logging.getLogger(__name__)
    return logger

def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


def calculate_global_latent_stats(vae, dataloader, device):
    vae = vae.to(device)
    vae.eval()
    total_sum = 0.0
    total_sq_sum = 0.0
    total_elements = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Statistics"): 
            raw_image, _ = batch
            raw_image = raw_image.to(device)
            latent,_,_ = vae.encode(raw_image)
            # latent = posterior.sample()
            
            flat_latent = latent.flatten()
            
            total_sum += flat_latent.sum().item()
            total_sq_sum += (flat_latent ** 2).sum().item()
            total_elements += flat_latent.numel()

    global_mean = total_sum / total_elements
    global_var = (total_sq_sum / total_elements) - (global_mean ** 2)
    global_std = max(torch.sqrt(torch.tensor(global_var)), 1e-6).item()

    return global_mean, global_std

#################################################################################
#                                  Training Loop                                #
#################################################################################

def main(args):    
    # set accelerator
    logging_dir = Path(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(
        project_dir=args.output_dir, logging_dir=logging_dir
        )

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)  # Make results folder (holds all experiment subfolders)

        model_string_name = args.model.replace("/", "-")  # e.g., DiT-XL/2 --> DiT-XL-2 (for naming folders)
        if args.vae_model is not None:
            vae_string_name = os.path.basename(args.vae_model) #args.vae_model
        else:
            vae_string_name = args.vae_config.split('/')[-2]
        if args.exp_index is not None:
            exp_index = int(args.exp_index)
        else:
            exp_index = len(glob(f"{args.output_dir}/*"))

        exp_name = f"{args.dataset}-{exp_index:03d}-{model_string_name}-{args.path_type}-{args.weighting}-{vae_string_name}-jepa{args.jepa_loss_weight}-blocktype{args.block_type}" 
        args.exp_name = exp_name
        
        save_dir = os.path.join(args.output_dir, args.exp_name)
        os.makedirs(save_dir, exist_ok=True)
        args_dict = vars(args)
        # Save to a JSON file
        json_dir = os.path.join(save_dir, "args.json")
        with open(json_dir, 'w') as f:
            json.dump(args_dict, f, indent=4)
        checkpoint_dir = f"{save_dir}/checkpoints"  # Stores saved model checkpoints
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(save_dir)
        logger.info(f"Experiment directory created at {save_dir}")
    else:
        model_string_name = args.model.replace("/", "-")  # e.g., DiT-XL/2 --> DiT-XL-2 (for naming folders)
        if args.vae_model is not None:
            vae_string_name = os.path.basename(args.vae_model) #args.vae_model
        else:
            vae_string_name = args.vae_config.split('/')[-2]
        if args.exp_index is not None:
            exp_index = int(args.exp_index)
        else:
            exp_index = len(glob(f"{args.output_dir}/*"))

        exp_name =  f"{args.dataset}-{exp_index:03d}-{model_string_name}-{args.path_type}-{args.weighting}-{vae_string_name}-jepa{args.jepa_loss_weight}-blocktype{args.block_type}" 
        args.exp_name = exp_name
        
        save_dir = os.path.join(args.output_dir, args.exp_name)
        # os.makedirs(save_dir, exist_ok=True)
        args_dict = vars(args)
        # Save to a JSON file
        # json_dir = os.path.join(save_dir, "args.json")
        # with open(json_dir, 'w') as f:
        #     json.dump(args_dict, f, indent=4)
        checkpoint_dir = f"{save_dir}/checkpoints"  # Stores saved model checkpoints
        # os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(save_dir)
        logger.info(f"Experiment directory created at {save_dir}")
        # logger = create_logger(None)
    device = accelerator.device
    if torch.backends.mps.is_available():
        accelerator.native_amp = False    
    if args.seed is not None:
        set_seed(args.seed + accelerator.process_index)
    
    # Create model:
    assert args.resolution % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."
    latent_size = args.resolution // 8
    
    # vae, vae_model, vae_embed_dim, dit_input_size, latents_bias, latents_scale, vae_1d = build_tokenizer(args.vae_config, args.vae_ckpt)
    # Create model:
    transform_mean = [0.5, 0.5, 0.5]
    transform_std = [0.5, 0.5, 0.5]
    if args.vae_model is not None:
        print(f"Loading VAE model: {args.vae_model}")
        # vae = SoftVQModel.from_pretrained(f"SoftVQVAE/{args.vae_model}")
        vae, vae_string_name, vae_embed_dim, dit_input_size, latents_bias, latents_scale, vae_1d = build_tokenizer(args.vae_config, args.vae_model)
                
        # vae_embed_dim = vae.codebook_embed_dim
        # vae_enc_patch_size = vae.encoder.patch_size
        # if vae.num_latent_tokens == 0:
        #     dit_input_size = (args.resolution // vae_enc_patch_size) ** 2
        #     vae_1d = False 
        # else:
        #     dit_input_size = vae.num_latent_tokens
        #     vae_1d = True
        # latents_bias = 0.0 # vae.vq_mean
        # latents_scale = 1.0 # vae.vq_std
        # vae_string_name = args.vae_model 
    else:
        raise NotImplementedError()
    print(f"vae_embed_dim: {vae_embed_dim}, dit_input_size: {dit_input_size}, vae_1d: {vae_1d}")
    vae.eval()
    

    z_dims = [0]
    block_kwargs = {"fused_attn": args.fused_attn, "qk_norm": args.qk_norm, "flash_attn": args.flash_attn}
    model = SiT_models[args.model](
        in_channels=vae_embed_dim,
        input_size=dit_input_size,
        num_classes=args.num_classes,
        use_cfg = (args.cfg_prob > 0),
        z_dims = z_dims,
        vae_1d=vae_1d,
        **block_kwargs
    )
        
    for param in vae.parameters():
        param.requires_grad = False

    ema = deepcopy(model) # Create an EMA of the model for use after training
    requires_grad(ema, False)
    
    # create loss function
    loss_fn = SILoss(
        prediction=args.prediction,
        path_type=args.path_type, 
        encoders=[],
        accelerator=accelerator,
        latents_scale=latents_scale,
        latents_bias=latents_bias,
        weighting=args.weighting
    )
    if accelerator.is_main_process:
        logger.info(f"SiT Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Setup optimizer (we used default Adam betas=(0.9, 0.999) and a constant learning rate of 1e-4 in our paper):
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )    
    
    # Setup data:
    # TODO: this is the feature dataset for ImageNet, you need to change this for your own dataset
    # train_dataset = CustomDataset(args.data_path)

    # transform = transforms.Compose([
    #     transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.resolution)),
    #     transforms.RandomHorizontalFlip(),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=transform_mean, std=transform_std, inplace=True)
    # ])
    transform = Compose([
    Resize((args.resolution, args.resolution)),
    ToTensor(),
    Normalize(mean=transform_mean, 
             std=transform_std, inplace=True)])
    if args.dataset == 'imagenet_cached':
        train_dataset = CachedFolder(args.data_path, transform=transform)
    else:
        train_dataset = ImageFolder(args.data_path, transform=transform)
    
    local_batch_size = int(args.batch_size // accelerator.num_processes)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=local_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    latents_bias, latents_scale = calculate_global_latent_stats(vae, train_dataloader, device)
    logger.info(f"latents_bias: {latents_bias} - latents_scale: {latents_scale}")
    if accelerator.is_main_process:
        logger.info(f"Dataset contains {len(train_dataset):,} images ({args.data_path})")
    
    update_ema(ema, model, decay=0)  # Ensure EMA is initialized with synced weights
    
    # resume:
    global_step = 0
    if args.resume_step > 0:
        ckpt_name = str(args.resume_step).zfill(7) +'.pt'
        ckpt = torch.load(
            f'{checkpoint_dir}/{ckpt_name}',
            map_location='cpu',
            )
        model_state_dict = {}
        for key, item in ckpt['model'].items():
            new_key = key.replace('module.', '')
            new_key = new_key.replace('_orig_mod.', '')
            model_state_dict[new_key] = item
        model.load_state_dict(model_state_dict)
        ema.load_state_dict(ckpt['ema'])
        optimizer.load_state_dict(ckpt['opt'])
        global_step = ckpt['steps']

    # torch compile
    if not args.disable_compile:
        model = torch.compile(model)
        vae = torch.compile(vae)
        
        
    vae = vae.to(device).eval()    
    decoder = vae.decoder
    model = model.to(device)
    ema = ema.to(device)  # Create an EMA of the model for use after training

    # Prepare models for training:
    model.train()  # important! This enables embedding dropout for classifier-free guidance
    ema.eval()  # EMA model should always be in eval mode

    model, optimizer, train_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader
    )

    if accelerator.is_main_process:
        tracker_config = vars(copy.deepcopy(args))
        accelerator.init_trackers(
            project_name="rule-tokenizer", 
            config=tracker_config,
            init_kwargs={
                "wandb": {"name": f"{args.exp_name}"}
            },
        )
        
    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    # TODO: change here
    # Labels to condition the model with (feel free to change):
    sample_batch_size = 64 // accelerator.num_processes
    ys = torch.randint(args.num_classes, size=(sample_batch_size,), device=device)
    ys = ys.to(device)
    # Create sampling noise:
    n = ys.size(0)
    xT = torch.randn((n, 4, latent_size, latent_size), device=device)
        
    for epoch in range(args.epochs):
        model.train()
        for data_batch in train_dataloader:
            
            if args.dataset == 'imagenet_cached':
                x, y = data_batch
                x = x.to(device, non_blocking=True)
                
            else:
                raw_image, y = data_batch
            
                raw_image = raw_image.to(device)

                with torch.no_grad() and accelerator.autocast():
                    # Map input images to latent space + normalize latents:
                    x, _, _ = vae.encode(raw_image)
                    qaunt_shape = x.shape

                
                if not vae_1d:
                    B, L, C = x.shape 
                    H, W = int(math.sqrt(L)), int(math.sqrt(L))
                    x = x.view(B, H, W, C).permute(0, 3, 1, 2)
            
            # print(x.shape)
                  
            y = y.to(device, non_blocking=True)
            z = None
            labels = y
            x = (x - latents_bias) / latents_scale

            with accelerator.accumulate(model):
                model_kwargs = dict(y=labels)
                loss, proj_loss = loss_fn(model, x, model_kwargs, zs=[])
                loss_mean = loss.mean()
                proj_loss_mean = proj_loss.mean()
                loss = loss_mean + proj_loss_mean * args.proj_coeff
                    
                ## optimization
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    params_to_clip = model.parameters()
                    grad_norm = accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

                if accelerator.sync_gradients:
                    update_ema(ema, model) # change ema function
            
            ### enter
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1                
            if (global_step % args.checkpointing_steps == 0 and global_step > 0) or global_step == args.max_train_steps:
                if accelerator.is_main_process:
                    checkpoint = {
                        "model": model.module.state_dict(),
                        "ema": ema.state_dict(),
                        "opt": optimizer.state_dict(),
                        "args": args,
                        "steps": global_step,
                    }
                    checkpoint_path = f"{checkpoint_dir}/{global_step:07d}.pt"
                    torch.save(checkpoint, checkpoint_path)
                    logger.info(f"Saved checkpoint to {checkpoint_path}")
                    manage_checkpoints(checkpoint_dir, keep_last_n=20)

            if (global_step == 1 or (global_step % args.sampling_steps == 0 and global_step > 0)) or global_step == args.max_train_steps:
                from modelling.samplers import euler_sampler, euler_maruyama_sampler

                model.eval()
                n = 64 // accelerator.num_processes
                if vae_1d:
                    quantT = torch.randn(n, dit_input_size, vae_embed_dim, device=device)
                else:
                    quantT = torch.randn(n, vae_embed_dim, dit_input_size, dit_input_size, device=device)
  
                # quantT = torch.rand(*qaunt_shape)
                ys = torch.randint(0, args.num_classes, (n,), device=device)
                # print(xT.shape,vae_embed_dim, dit_input_size)
                with torch.no_grad():
                    quant = euler_sampler(
                        model, 
                        quantT, 
                        ys,
                        num_steps=50, 
                        cfg_scale=4.0,
                        guidance_low=0.,
                        guidance_high=1.,
                        path_type=args.path_type,
                        heun=False,
                        num_classes = args.num_classes
                    ).to(torch.float32)
                    quant = quant * latents_scale + latents_bias
                    xT = torch.randn(quant.size(0), 3, args.resolution, args.resolution, device=quant.device)
                    # print("quant.shape:",quant.shape,"xT.shape:",xT.shape)
                    B, C, H, W = quant.shape  
                    quant = quant.view(B, C, H * W).permute(0, 2, 1) 
                    # print("quant.shape:",quant.shape,"xT.shape:",xT.shape)
                    samples = euler_sampler(
                        decoder, 
                        xT, 
                        quant,
                        num_steps=50, 
                        cfg_scale=1.0,
                        guidance_low=0.,
                        guidance_high=1.,
                        path_type=args.path_type,
                        heun=False,
                        num_classes = args.num_classes
                    ).to(torch.float32)
                    # if not vae_1d:
                    #     B, C, H, W = samples.shape
                    #     samples = samples.view(B, C, H * W).permute(0, 2, 1)

                    # samples = vae.decode(samples * latents_scale + latents_bias)                    
                    samples = torch.clamp(samples, -1.0, 1.0)
                    samples = (samples + 1) / 2
                    # samples = samples.permute(1, 2, 0).mul_(255).cpu().numpy()
                    # print("samples.shape:",samples.shape)

    
                    
                out_samples = accelerator.gather(samples.to(torch.float32))
                print("out_samples.shape:",out_samples.shape)
                accelerator.log({"samples": wandb.Image(array2grid(out_samples)),
                                 })
                logging.info("Generating EMA samples done.")
                model.train()

            logs = {
                "loss": accelerator.gather(loss_mean).mean().detach().item(), 
                "proj_loss": accelerator.gather(proj_loss_mean).mean().detach().item(),
                "grad_norm": accelerator.gather(grad_norm).mean().detach().item()
            }
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break
        if global_step >= args.max_train_steps:
            break

    model.eval()  # important! This disables randomized embedding dropout
    # do any sampling/FID calculation/etc. with ema (or model) in eval mode ...
    
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        logger.info("Done!")
    accelerator.end_training()


def manage_checkpoints(save_dir, keep_last_n=10):
    # List all checkpoint files in the save directory
    checkpoints = [f for f in os.listdir(save_dir) if f.endswith('.pt')]
    checkpoints.sort(key=lambda f: int(f.split('/')[-1].split('.')[0]))  # Sort by epoch number

    # If more than `keep_last_n` checkpoints exist, remove the oldest ones
    if len(checkpoints) > keep_last_n + 1:  # keep_last_n + 1 to account for the latest checkpoint
        for checkpoint_file in checkpoints[:-keep_last_n-1]:
            checkpoint_path = os.path.join(save_dir, checkpoint_file)
            if os.path.exists(checkpoint_path):
                os.remove(checkpoint_path)
                print(f"Removed old checkpoint: {checkpoint_path}")


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Training")

    # logging:
    parser.add_argument("--output-dir", type=str, default="exps")
    parser.add_argument("--logging-dir", type=str, default="logs")
    parser.add_argument("--report-to", type=str, default="wandb")
    parser.add_argument("--resume-step", type=int, default=0)

    # model
    parser.add_argument("--model", type=str)
    parser.add_argument("--num-classes", type=int, default=4)
    parser.add_argument("--fused-attn", action='store_true', default=False)
    parser.add_argument("--flash-attn",  default=True)
    parser.add_argument("--qk-norm",  action='store_true', default=False)

    # dataset
    parser.add_argument("--dataset", type=str, default="imagenet", choices=["imagenet", "imagenet_cached"])
    parser.add_argument("--data-path", type=str, default="/cpfs04/user/hanyujin/causal-dm/synthetic_data_number50000_size64")
    parser.add_argument("--resolution", type=int, choices=[64, 256], default=64)
    parser.add_argument("--batch-size", type=int, default=256)

    # precision
    parser.add_argument("--allow-tf32", action="store_true")
    parser.add_argument("--mixed-precision", type=str, default="fp16", choices=["no", "fp16", "bf16"])

    # optimization
    # parser.add_argument("--epochs", type=int, default=1000)
    # parser.add_argument("--max-train-steps", type=int, default=200000)
    # parser.add_argument("--checkpointing-steps", type=int, default=40000)
    # parser.add_argument("--sampling-steps", type=int, default=5000)
    parser.add_argument("--epochs", type=int, default=600)
    parser.add_argument("--max-train-steps", type=int, default=120000)
    parser.add_argument("--checkpointing-steps", type=int, default=10000)
    parser.add_argument("--sampling-steps", type=int, default=1000)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--adam-beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam-beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam-weight-decay", type=float, default=0., help="Weight decay to use.")
    parser.add_argument("--adam-epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max-grad-norm", default=1.0, type=float, help="Max gradient norm.")

    # seed
    parser.add_argument("--seed", type=int, default=0)

    # cpu
    parser.add_argument("--num-workers", type=int, default=8)

    # loss
    parser.add_argument("--path-type", type=str, default="linear", choices=["linear", "cosine"])
    parser.add_argument("--prediction", type=str, default="v", choices=["v"]) # currently we only support v-prediction
    parser.add_argument("--cfg-prob", type=float, default=0.1)
    parser.add_argument("--enc-type", type=str, default='dinov2-vit-b')
    parser.add_argument("--proj-coeff", type=float, default=0.5)
    parser.add_argument("--weighting", default="uniform", type=str, help="Max gradient norm.") # lognormal
    parser.add_argument("--legacy", action='store_true', default=False)
    
    parser.add_argument("--vae-model", default="softvq-l-64") 
    parser.add_argument("--exp-index", type=str, default=None, help="experiment index")
    parser.add_argument('--vae-config', default="pretrained_models/vqvae2/imagenet256", type=str)
    # parser.add_argument('--vae-ckpt', default="pretrained_models/vqvae2/imagenet256", type=str)
    
    parser.add_argument("--disable-compile", action='store_true', default=False)
    
    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()
    
    if args.vae_config is not None:
        with open(args.vae_config, 'r', encoding='utf-8') as f:
            file_yaml = yaml.YAML()
            config_args = file_yaml.load(f)
            for key, value in config_args.items():
                if key=='jepa_loss_weight' or key=='block_type' or key=='dataset' or key =="data_path":
                    vars(args)[key] = value

    return args

if __name__ == "__main__":
    args = parse_args()
    
    main(args)
