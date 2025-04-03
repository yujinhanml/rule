import yaml
import sys
sys.path.append('/cpfs04/user/hanyujin/rule-gen/rule_tokenizer')
import torch
from modelling.tokenizer import VQ_models



def build_tokenizer(vq_config,
                    vq_ckpt):
    
    with open(vq_config, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    config_name = vq_config.split('/')[-2]
    # exp_index = int(config_name.split('-')[0][3:])

    vae = VQ_models[config['vq_model']](
        image_size=config['image_size'],
        codebook_size=16384, #config['codebook_size'],
        codebook_embed_dim=config['codebook_embed_dim'],
        enc_type=config['enc_type'],
        encoder_model=config['encoder_model'],
        dec_type=config['dec_type'],
        decoder_model=config['decoder_model'],
        num_latent_tokens=config['num_latent_tokens'],
        enc_tuning_method=config['encoder_tuning_method'],
        dec_tuning_method=config['decoder_tuning_method'],
        enc_patch_size=config['encoder_patch_size'],
        dec_patch_size=config['decoder_patch_size'],
        tau=config['tau'] if 'tau' in config else 1.0,
        repa=False,
    )

    # vq_model.to(device)
    # vq_model.eval()
    checkpoint = torch.load(vq_ckpt, map_location="cpu", weights_only=False)
    model_weight = checkpoint['model']
    # if "ema" in checkpoint:  # ema
    #     model_weight = checkpoint["ema"]
    # elif "model" in checkpoint:  # ddp
    #     model_weight = checkpoint["model"]
    # elif "state_dict" in checkpoint:
    #     model_weight = checkpoint["state_dict"]
    # else:
    #     raise Exception("please check model weight")
    keys = vae.load_state_dict(model_weight, strict=False)
    print(keys)
    

    vq_1d = False
    # if config_name == 'exp003-aejepa-16':
    vq_mean, vq_std = 0.0, 1.0
    dit_input_size = 16
    # else:
    #     raise NotImplementedError
    
    
    return vae, config['vq_model'], config['codebook_embed_dim'], dit_input_size, vq_mean, vq_std, vq_1d