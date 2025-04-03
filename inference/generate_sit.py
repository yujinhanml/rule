# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Samples a large number of images from a pre-trained SiT model using DDP.
Subsequently saves a .npz file that can be used to compute FID and other
evaluation metrics via the ADM repo: https://github.com/openai/guided-diffusion/tree/main/evaluations

For a simple single-GPU/CPU sampling script, see sample.py.
"""
import torch
import torch.distributed as dist
from tqdm import tqdm
import os
import sys
sys.path.append('../')
import ruamel.yaml as yaml
from PIL import Image
import numpy as np
import math
import argparse
from modelling.sit import SiT_models, SiT
from modelling.samplers import euler_sampler, euler_maruyama_sampler
from modelling.tokenizer import SoftVQModel
from utils.model import build_tokenizer

def create_npz_from_sample_folder(sample_dir, num=50_000):
    """
    Builds a single .npz file from a folder of .png samples.
    """
    samples = []
    for i in tqdm(range(num), desc="Building .npz file from samples"):
        sample_pil = Image.open(f"{sample_dir}/{i:06d}.png")
        sample_np = np.asarray(sample_pil).astype(np.uint8)
        samples.append(sample_np)
    samples = np.stack(samples)
    assert samples.shape == (num, samples.shape[1], samples.shape[2], 3)
    npz_path = f"{sample_dir}.npz"
    np.savez(npz_path, arr_0=samples)
    print(f"Saved .npz file to {npz_path} [shape={samples.shape}].")
    return npz_path


def main(args):
    """
    Run sampling.
    """
    torch.backends.cuda.matmul.allow_tf32 = args.tf32  # True: fast but may lead to some small numerical differences
    assert torch.cuda.is_available(), "Sampling with DDP requires at least one GPU. sample.py supports CPU-only usage"
    torch.set_grad_enabled(False)

    # Setup DDP:cd
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")
    
    if args.vae_model is not None:
        print(f"Loading VAE model: {args.vae_model}")
        # vae = SoftVQModel.from_pretrained(f"SoftVQVAE/{args.vae_model}")
        vae, vae_string_name, vae_embed_dim, dit_input_size, latents_bias, latents_scale, vae_1d = build_tokenizer(args.vae_config, args.vae_model)
        # vae_embed_dim = vae.codebook_embed_dim
        # dit_input_size = vae.num_latent_tokens
        latents_bias = args.vq_mean #-0.23420482625881933 #vae.vq_mean
        latents_scale = args.vq_std #1.6881186962127686 #vae.vq_std
        vae_string_name = os.path.basename(args.vae_model) #args.vae_model
        # vae_1d = True
    else:
        raise NotImplementedError()
    print(f"vae_embed_dim: {vae_embed_dim}, dit_input_size: {dit_input_size}, vae_1d: {vae_1d}")
    vae = torch.compile(vae)
    vae = vae.to(device)
    vae.eval()
    
    # Auto-download a pre-trained model or load a custom SiT checkpoint from train.py:
    # if args.vae_model is not None:
    #     model = SiT.from_pretrained(f"SoftVQVAE/sit-xl_{args.vae_model}")
    #     model = torch.compile(model)
    #     args.model = "SiT-XL/1"
    # else:
    ckpt_path = args.ckpt 
    # Load model:
    block_kwargs = {"fused_attn": args.fused_attn, 'flash_attn':args.flash_attn, "qk_norm": args.qk_norm}
    z_dims = [0]
    model = SiT_models[args.model](
        in_channels=vae_embed_dim,
        input_size=dit_input_size,
        num_classes=args.num_classes,
        use_cfg = True,
        z_dims = z_dims, #[int(z_dim) for z_dim in args.projector_embed_dims.split(',')],
        # encoder_depth=args.encoder_depth,
        vae_1d=vae_1d,
        **block_kwargs,
    )
    state_dict = torch.load(ckpt_path, map_location=f'cuda:{device}')['ema']
    # state_dict = torch.load(ckpt_path, map_location=f'cuda:{device}')['model']
    compiled_state_dict = False 
    for key, item in state_dict.items():
        if '_orig_mod' in key:
            compiled_state_dict = True
            break
    if compiled_state_dict:
        model = torch.compile(model)
        keys = model.load_state_dict(state_dict, strict=False)
    else:
        keys = model.load_state_dict(state_dict, strict=False)
        model = torch.compile(model)
    print(keys)
    model = model.to(device)
    model.eval()  # important!
    # vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)
    
    
    assert args.cfg_scale >= 1.0, "In almost all cases, cfg_scale be >= 1.0"
    using_cfg = args.cfg_scale > 1.0

    # Create folder to save samples:
    model_string_name = args.model.replace("/", "-")
    # vae_string_name = args.vae_config.split('/')[-2]
    folder_name = f"{args.vq_model}-{args.dataset}-{model_string_name}-{args.path_type}-vae{vae_string_name}-size-{args.resolution}" \
                  f"cfg-{args.cfg_scale}-seed-{args.global_seed}"
    sample_folder_dir = f"{args.sample_dir}/{folder_name}"
    if rank == 0:
        os.makedirs(sample_folder_dir, exist_ok=True)
        print(f"Saving .png samples at {sample_folder_dir}")
    dist.barrier()

    # Figure out how many samples we need to generate on each GPU and how many iterations we need to run:
    n = args.per_proc_batch_size
    global_batch_size = n * dist.get_world_size()
    # To make things evenly-divisible, we'll sample a bit more than we need and then discard the extra samples:
    total_samples = int(math.ceil(args.num_fid_samples / global_batch_size) * global_batch_size)
    if rank == 0:
        print(f"Total number of images that will be sampled: {total_samples}")
        print(f"SiT Parameters: {sum(p.numel() for p in model.parameters()):,}")
    assert total_samples % dist.get_world_size() == 0, "total_samples must be divisible by world_size"
    samples_needed_this_gpu = int(total_samples // dist.get_world_size())
    assert samples_needed_this_gpu % n == 0, "samples_needed_this_gpu must be divisible by the per-GPU batch size"
    iterations = int(samples_needed_this_gpu // n)
    pbar = range(iterations)
    pbar = tqdm(pbar) if rank == 0 else pbar
    total = 0
    ptdtype = {'none': torch.float32, 'bf16': torch.bfloat16, 'fp16': torch.float16}[args.mixed_precision]
    
    for i in pbar:
        # Sample inputs:
        with torch.cuda.amp.autocast(dtype=ptdtype): 
            # if vae_1d:
            #     z = torch.randn(n, dit_input_size, vae_embed_dim, device=device)
            # else:
            #     z = torch.randn(n, vae_embed_dim, dit_input_size, dit_input_size, device=device)
            # y = torch.randint(0, args.num_classes, (n,), device=device)

            # Setup classifier-free guidance:
            # if using_cfg:
            #     z = torch.cat([z, z], 0)
            #     y_null = torch.tensor([args.num_classes] * n, device=device)
            #     y = torch.cat([y, y_null], 0)
            
            # Sample Latents
            if vae_1d:
                quantT = torch.randn(n, dit_input_size, vae_embed_dim, device=device)
            else:
                quantT = torch.randn(n, vae_embed_dim, dit_input_size, dit_input_size, device=device)
            if using_cfg:
                ys = torch.randint(0, args.num_classes, (n,), device=device)
            # Sample images:
            sampling_kwargs_lat = dict(
                model=model, 
                latents=quantT,
                y=ys,
                num_steps=args.num_steps, 
                heun=args.heun,
                cfg_scale=args.cfg_scale,
                guidance_low=args.guidance_low,
                guidance_high=args.guidance_high,
                path_type=args.path_type,
                num_classes =  args.num_classes
            )
            with torch.no_grad():
                if args.mode == "sde":
                    quant = euler_maruyama_sampler(**sampling_kwargs_lat).to(torch.float32)
                    quant = quant * latents_scale + latents_bias
                    xT = torch.randn(quant.size(0), 3, args.resolution, args.resolution, device=quant.device)
                    B, C, H, W = quant.shape  
                    quant = quant.view(B, C, H * W).permute(0, 2, 1) 
                    sampling_kwargs_img = dict(
                        model=vae.decoder, 
                        latents=xT,
                        y=quant,
                        num_steps=args.num_steps, 
                        heun=args.heun,
                        cfg_scale=1,
                        guidance_low=args.guidance_low,
                        guidance_high=args.guidance_high,
                        path_type=args.path_type,
                        num_classes =  args.num_classes
                    )
                    samples = euler_maruyama_sampler(**sampling_kwargs_img).to(torch.float32)
                elif args.mode == "ode":
                    quant = euler_sampler(**sampling_kwargs_lat).to(torch.float32)
                    quant = quant * latents_scale + latents_bias
                    xT = torch.randn(quant.size(0), 3, args.resolution, args.resolution, device=quant.device)
                    B, C, H, W = quant.shape  
                    quant = quant.view(B, C, H * W).permute(0, 2, 1) 
                    sampling_kwargs_img = dict(
                        model=vae.decoder, 
                        latents=xT,
                        y=quant,
                        num_steps=args.num_steps, 
                        heun=args.heun,
                        cfg_scale=1,
                        guidance_low=args.guidance_low,
                        guidance_high=args.guidance_high,
                        path_type=args.path_type,
                        num_classes =  args.num_classes
                    )
                    samples = euler_sampler(**sampling_kwargs_img).to(torch.float32)
                else:
                    raise NotImplementedError()
                # if using_cfg:
                #     samples, _ = samples.chunk(2, dim=0)  # Remove null class samples
                # if not vae_1d:
                #     B, C, H, W = samples.shape
                #     samples = samples.view(B, C, H * W).permute(0, 2, 1)

                # samples = vae.decode(samples * latents_scale + latents_bias)
                samples = torch.clamp(samples, -1.0, 1.0)
                samples = (samples + 1) / 2
                    
            samples = torch.clamp(
                255. * samples, 0, 255
                ).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()
            # print("samples.shape:",samples.shape)
            # Save samples to disk as individual .png files
            for i, sample in enumerate(samples):
                index = i * dist.get_world_size() + rank + total
                # print("sample.shape:",sample.shape)
                Image.fromarray(sample).save(f"{sample_folder_dir}/{index:06d}.png")
        total += global_batch_size

    # Make sure all processes have finished saving their samples before attempting to convert to .npz
    dist.barrier()
    if rank == 0:
        create_npz_from_sample_folder(sample_folder_dir, args.num_fid_samples)
        print("Done.")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # seed
    parser.add_argument("--global-seed", type=int, default=0)

    # precision
    parser.add_argument("--tf32", type=bool, default=False,
                        help="By default, use TF32 matmuls. This massively accelerates sampling on Ampere GPUs.")

    # logging/saving:
    parser.add_argument("--ckpt", type=str, default=None, help="Optional path to a SiT checkpoint.")
    parser.add_argument("--sample-dir", type=str, default="/cpfs04/user/hanyujin/rule-gen/experiments/samples")

    # model
    parser.add_argument("--model", type=str, choices=list(SiT_models.keys()), default="SiT-XL/2")
    parser.add_argument("--num-classes", type=int, default=4)
    parser.add_argument("--resolution", type=int, choices=[64,256, 512], default=64)
    parser.add_argument("--fused-attn", type=bool, default=False)
    parser.add_argument("--flash-attn",  type=bool, default=True)
    parser.add_argument("--qk-norm",  type=bool, default=False)
    parser.add_argument("--vq_mean",  type=float, default=0.0)
    parser.add_argument("--vq_std",  type=float, default=1.0)


    # number of samples
    parser.add_argument("--per-proc-batch-size", type=int, default=64)
    parser.add_argument("--num-fid-samples", type=int, default=1000)

    # sampling related hyperparameters
    parser.add_argument("--mode", type=str, default="ode")
    parser.add_argument("--cfg-scale",  type=float, default=1.5)
    parser.add_argument("--path-type", type=str, default="linear", choices=["linear", "cosine"])
    parser.add_argument("--num-steps", type=int, default=50)
    parser.add_argument("--heun", type=bool,  default=False) # only for ode
    parser.add_argument("--guidance-low", type=float, default=0.)
    parser.add_argument("--guidance-high", type=float, default=1.)

    # will be deprecated
    parser.add_argument("--legacy", type=bool, default=False) # only for ode

    parser.add_argument("--vae-model", type=str, default="softvq-l-64") 
    parser.add_argument('--vae-config', default="pretrained_models/vqvae2/imagenet256", type=str)
    # parser.add_argument('--vae-ckpt', default="pretrained_models/vqvae2/imagenet256", type=str)

    parser.add_argument("--mixed-precision", type=str, default='fp16', choices=["none", "fp16", "bf16"]) 
    
    args = parser.parse_args()
    if args.vae_config is not None:
        with open(args.vae_config, 'r', encoding='utf-8') as f:
            file_yaml = yaml.YAML()
            config_args = file_yaml.load(f)
            for key, value in config_args.items():
                if key=='dataset' or key =="data_path" or key =="vq_model":
                    vars(args)[key] = value
                # if key =="data_path":
                #     vars(args)["data_path"] = value
    # args.dataset = args.vae_config['dataset']
    main(args)
