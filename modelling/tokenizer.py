# Modified from:
#   taming-transformers: https://github.com/CompVis/taming-transformers
#   maskgit: https://github.com/google-research/maskgit
from dataclasses import dataclass, field
from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import PyTorchModelHubMixin

import sys
sys.path.append('/home/haoc/storage/rule_tokenizer')
from modelling.modules import Encoder, Decoder, TimmViTEncoder, TimmViTDecoder
from modelling.quantizers.vq import VectorQuantizer
from modelling.quantizers.kl import DiagonalGaussianDistribution
from modelling.quantizers.softvq import SoftVectorQuantizer
from modelling.modules.jepa_vit.vision_transformer import vit_predictor

from modelling.sit_decoder import SiT_models
from losses.sit_loss import SILoss
from modelling.samplers import euler_maruyama_sampler

from timm import create_model

from masks.utils import apply_masks


def mean_flat(x):
    """
    Take the mean over all non-batch dimensions.
    """
    return torch.mean(x, dim=list(range(1, len(x.size()))))


def build_mlp(hidden_size, projector_dim, z_dim):
    return nn.Sequential(
                nn.Linear(hidden_size, projector_dim),
                nn.SiLU(),
                nn.Linear(projector_dim, projector_dim),
                nn.SiLU(),
                nn.Linear(projector_dim, z_dim),
            )


@dataclass
class ModelArgs:
    image_size: int = 256
    base_image_size: int = 256
    
    codebook_size: int = 16384
    codebook_embed_dim: int = 8
    codebook_l2_norm: bool = True
    codebook_show_usage: bool = True
    commit_loss_beta: float = 0.25
    entropy_loss_ratio: float = 0.0
    vq_loss_ratio: float = 1.0 # for soft vq
    kl_loss_weight: float = 0.000001
    tau: float = 0.1
    num_codebooks: int = 1
    
    encoder_ch_mult: List[int] = field(default_factory=lambda: [1, 1, 2, 2, 4])
    decoder_ch_mult: List[int] = field(default_factory=lambda: [1, 1, 2, 2, 4])
    z_channels: int = 256
    dropout_p: float = 0.0

    enc_type: str = 'cnn'
    dec_type: str = 'cnn'
    encoder_model: str = 'llamagen_encoder'
    decoder_model: str = 'llamagen_decoder'
    num_latent_tokens: int = 256
    to_pixel: str = 'linear'
    
    # for pre-trained models
    enc_tuning_method: str = 'full'
    dec_tuning_method: str = 'full'
    enc_pretrained: bool = True
    dec_pretrained: bool = False 
    
    # for vit 
    enc_patch_size: int = 16
    dec_patch_size: int = 16
    enc_drop_path_rate: float = 0.0
    dec_drop_path_rate: float = 0.0

    # encoder token drop
    # NOTE: not used
    enc_token_drop: float = 0.1
    enc_token_drop_max: float = 0.1
    
    # deocder cls token
    enc_cls_token: bool = True
    dec_cls_token: bool = True
    
    # rope
    use_ape: bool = True 
    use_rope: bool = False
    rope_mixed: bool = False
    rope_theta: float = 10.0
    
    # repa for vit
    repa: bool = False
    repa_patch_size: int = 16
    repa_model: str = 'vit_base_patch16_224'
    repa_proj_dim: int = 2048
    repa_loss_weight: float = 0.1
    repa_align: str = 'global'
    
    # jepa 
    predictor_depth: int = 12
    predictor_embed_dim: int = 384
    
    
    vq_mean: float = 0.0
    vq_std: float = 1.0
    

class VQModel(nn.Module, PyTorchModelHubMixin):
    def __init__(self, config: ModelArgs, 
                tags=["arxiv:2412.10958", "image-generation", "32 tokens", "SoftVQ-VAE"], 
                repo_url="https://github.com/Hhhhhhao/continuous_tokenizer", 
                license="apache-2.0"):
        super().__init__()
        self.config = config
        self.vq_mean = config.vq_mean
        self.vq_std = config.vq_std
        self.num_latent_tokens = config.num_latent_tokens
        self.codebook_embed_dim = config.codebook_embed_dim
        
        self.repa = config.repa
        self.repa_loss_weight = config.repa_loss_weight
        self.repa_align = config.repa_align
        if config.repa and config.enc_type == 'vit':
            self.repa_model = create_model(config.repa_model, pretrained=True, img_size=config.image_size, patch_size=config.repa_patch_size)
            for param in self.repa_model.parameters():
                param.requires_grad = False
            self.repa_model.eval()
            repa_z_dim = self.repa_model.embed_dim
            self.repa_z_dim = repa_z_dim
            self.projection = build_mlp(config.codebook_embed_dim, config.repa_proj_dim, repa_z_dim)
            from modelling.lpips.lpips_timm import Normalize, Denormalize
            self.de_scale = Denormalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            self.scale = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        else:
            repa_z_dim = None
        
        
        if config.enc_type == 'cnn':
            if config.encoder_model == 'llamagen_encoder':
                self.encoder = Encoder(ch_mult=config.encoder_ch_mult, z_channels=config.z_channels, dropout=config.dropout_p)
            else:
                raise NotImplementedError
            self.quant_conv = nn.Conv2d(config.z_channels, config.codebook_embed_dim, 1)
        elif config.enc_type == 'vit':
            self.encoder = TimmViTEncoder(
                in_channels=3, num_latent_tokens=config.num_latent_tokens,
                model_name=config.encoder_model,  # 'vit_small_patch14_dinov2.lvd142m', #'vit_base_patch14_dinov2.lvd142m',  #
                model_kwargs={'img_size': config.image_size, 'patch_size': config.enc_patch_size, 'drop_path_rate': config.enc_drop_path_rate},
                pretrained=config.enc_pretrained,
                tuning_method=config.enc_tuning_method,
                tuning_kwargs={'r': 8},
                use_ape=config.use_ape, use_rope=config.use_rope, rope_mixed=config.rope_mixed, rope_theta=config.rope_theta,
                token_drop=config.enc_token_drop,
                token_drop_max=config.enc_token_drop_max,
                base_img_size=config.base_image_size,
                cls_token=config.enc_cls_token,
            )
            self.quant_conv = nn.Linear(self.encoder.embed_dim, config.codebook_embed_dim)
            
        
        if config.dec_type == 'cnn':
            if config.decoder_model == 'llamagen_decoder':
                self.decoder = Decoder(ch_mult=config.decoder_ch_mult, z_channels=config.z_channels, dropout=config.dropout_p)
            else:
                raise NotImplementedError
            self.post_quant_conv = nn.Conv2d(config.codebook_embed_dim, config.z_channels, 1)
        elif config.dec_type == 'vit':
            self.decoder = TimmViTDecoder(
                in_channels=3, num_latent_tokens=config.num_latent_tokens,
                model_name=config.decoder_model,  # 'vit_small_patch14_dinov2.lvd142m', #'vit_base_patch14_dinov2.lvd142m',  #
                model_kwargs={'img_size': config.image_size, 'patch_size': config.dec_patch_size, 'drop_path_rate': config.dec_drop_path_rate, 'latent_dim': config.codebook_embed_dim},
                pretrained=config.dec_pretrained,
                tuning_method=config.dec_tuning_method,
                tuning_kwargs={'r': 8},
                use_ape=config.use_ape, use_rope=config.use_rope, rope_mixed=config.rope_mixed, rope_theta=config.rope_theta,
                cls_token=config.dec_cls_token,
                codebook_embed_dim=config.codebook_embed_dim,
                to_pixel=config.to_pixel,
                base_img_size=config.base_image_size
            )
            self.post_quant_conv = nn.Linear(config.codebook_embed_dim, self.decoder.embed_dim)
        # check movq
        if 'movq' in config.decoder_model:
            self.use_movq = True 
        else:
            self.use_movq = False
        
        
        self.quantize = VectorQuantizer(config.codebook_size, config.codebook_embed_dim, 
                                        config.commit_loss_beta, config.entropy_loss_ratio,
                                        config.codebook_l2_norm, config.codebook_show_usage)

    def encode(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        quant, emb_loss, info = self.quantize(h)
        
        if self.repa and self.training:
            # get z from repa_encoder
            rescale_x = self.scale(self.de_scale(x))
            z = self.repa_model.forward_features(rescale_x)[:, self.repa_model.num_prefix_tokens:]

            # taking average over spatial dimension
            if self.repa_align == 'global':
                z = z.mean(dim=1)
                z_hat = quant.mean(dim=1)
                # calculate repa loss
                z_hat = self.projection(z_hat)
            elif self.repa_align == 'avg_1d':
                z = F.adaptive_avg_pool1d(z.permute(0, 2, 1), quant.shape[1]).permute(0, 2, 1)
                z_hat = quant
                z_hat = self.projection(z_hat)
            elif self.repa_align == 'avg_1d_shuffle':
                # shuffle the length dimension of z and avg
                indices = torch.randperm(z.shape[1])
                z = F.adaptive_avg_pool1d(z[:, indices, :].permute(0, 2, 1) , quant.shape[1]).permute(0, 2, 1)
                z_hat = quant
                z_hat = self.projection(z_hat)
            elif self.repa_align == 'repeat':
                z_hat = self.projection(quant)
                b, l, d = z_hat.shape
                z_hat = z_hat.unsqueeze(2).expand(-1, -1, z.size(1) // l, -1).reshape(b, -1, d)
            

            z = F.normalize(z, dim=-1)
            z_hat = F.normalize(z_hat, dim=-1)
            proj_loss = mean_flat(-(z * z_hat).sum(dim=-1))
            proj_loss = proj_loss.mean()
            proj_loss *= self.repa_loss_weight
            
            emb_loss += (proj_loss,)
        
        return quant, emb_loss, info

    def decode(self, quant, x=None, h=None, w=None):
        tmp_quant = quant 
        quant = self.post_quant_conv(quant)
        if self.use_movq:
            dec = self.decoder(quant, tmp_quant, h, w)
        else:
            dec = self.decoder(quant, None, h, w)
        return dec

    def decode_code(self, code_b, shape=None, channel_first=True):
        quant_b = self.quantize.get_codebook_entry(code_b, shape, channel_first)
        dec = self.decode(quant_b)
        return dec

    def forward(self, input):
        b, _, h, w = input.size()
        quant, diff, info = self.encode(input)
        self.quant = quant
        dec = self.decode(quant, x=input, h=h, w=w)
        return dec, diff, info


class SoftVQModel(VQModel, PyTorchModelHubMixin):
    def __init__(self, config: ModelArgs, 
                tags=["arxiv:2412.10958", "image-generation", "32 tokens", "SoftVQ-VAE"], 
                repo_url="https://github.com/Hhhhhhao/continuous_tokenizer", 
                license="apache-2.0"):
        super().__init__(config)
        self.quantize = SoftVectorQuantizer(config.codebook_size, config.codebook_embed_dim, 
                                            config.entropy_loss_ratio, 
                                            config.tau,                                   
                                            config.num_codebooks,
                                            config.codebook_l2_norm, config.codebook_show_usage)


class KLModel(VQModel):
    def __init__(self, config: ModelArgs):
        super().__init__(config)
        self.kl_loss_weight = config.kl_loss_weight
        self.quantize = None
        
        if config.enc_type == 'cnn':
            self.quant_conv = nn.Conv2d(config.z_channels, config.codebook_embed_dim * 2, 1)
        elif config.enc_type == 'vit':
            self.quant_conv = nn.Linear(self.encoder.embed_dim, config.codebook_embed_dim * 2)
        

    def encode(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        # quant, emb_loss, info = self.quantize(h)
        h_posterior = DiagonalGaussianDistribution(h)
        return h_posterior

    def decode(self, z):
        z = self.post_quant_conv(z)
        dec = self.decoder(z)
        return dec

    def decode_code(self, posterior, shape=None):
        z = posterior.sample()
        dec = self.decode(z)
        return dec

    def forward(self, input, sample_posterior=True):
        posterior = self.encode(input)
        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()
        dec = self.decode(z)
        # compute kl loss
        kl_loss = posterior.kl()
        kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]
        diff = (kl_loss * self.kl_loss_weight, torch.tensor(0.), torch.tensor(0.), torch.tensor(0.))
        return dec, diff, None


class AEModel(VQModel):
    def __init__(self, config: ModelArgs,
                tags=["arxiv:xxx", "image-generation", "1d-tokenizer", "128 tokens", "MAETok"], 
                repo_url="https://github.com/Hhhhhhao/continuous_tokenizer", 
                license="apache-2.0"):
        super().__init__(config)
        self.quantize = None 


    def encode(self, x):
        
        h = self.encoder(x)
        quant = self.quant_conv(h)
        emb_loss = (torch.tensor(0.), torch.tensor(0.), torch.tensor(0.), torch.tensor(0.))
        info = None
        return quant, emb_loss, info

    def decode(self, ):
        z = self.post_quant_conv(z)
        dec = self.decoder(z)
        return dec

    def decode(self, quant, x=None, h=None, w=None):
        tmp_quant = quant 
        quant = self.post_quant_conv(quant)
        if self.use_movq:
            dec = self.decoder(quant, tmp_quant, h, w)
        else:
            dec = self.decoder(quant, None, h, w)
        return dec
    

class AEJEPAModel(VQModel):
    def __init__(self, config: ModelArgs):
        super().__init__(config)
        self.quantize = None 
        self.predictor = vit_predictor(
            num_patches=self.encoder.num_img_tokens,
            num_heads=self.encoder.model.num_heads,
            depth=config.predictor_depth,
            predictor_embed_dim=config.predictor_embed_dim,
            # linear predict embed dim
            embed_dim=config.codebook_embed_dim,
            num_latent_tokens=config.num_latent_tokens,
        )
        self.encoder.model.dynamic_img_size = True

    def encode(self, x, masks_enc=None, return_img_tokens=False):
        
        h = self.encoder(x, masks_enc, return_img_tokens)
        quant = self.quant_conv(h)
        emb_loss = (torch.tensor(0.), torch.tensor(0.), torch.tensor(0.), torch.tensor(0.))
        info = None
        return quant, emb_loss, info

    def decode(self, quant, x=None, h=None, w=None, masks_enc=None):
        tmp_quant = quant 
        quant = self.post_quant_conv(quant)
        if self.use_movq:
            dec = self.decoder(quant, tmp_quant, h, w, masks_enc)
        else:
            dec = self.decoder(quant, None, h, w, masks_enc)
        return dec
    

    def forward(self, input, masks_enc=None, masks_pred=None):
        b, _, h, w = input.size()
        quant, diff, info = self.encode(input, masks_enc)
        
        self.quant = quant
        dec = self.decode(quant, x=input, h=h, w=w, masks_enc=masks_enc)
        
        # jepa related
        if self.training:
            # predictor
            # quant = apply_masks(quant, masks_enc)
            pred_z = self.predictor(quant, masks_enc, masks_pred)
            
            info = {'pred_z': pred_z}
        return dec, diff, info


class AEDiffModel(AEModel):
    def __init__(self, config: ModelArgs):
        super().__init__(config)
        block_kwargs = {"fused_attn": False, "qk_norm": False, "flash_attn": True}
        self.decoder = SiT_models[config.decoder_model](
                image_size=config.image_size,
                patch_size=config.dec_patch_size,
                in_channels=3,
                latent_channels=config.codebook_embed_dim,
                num_latent_tokens=config.num_latent_tokens if config.num_latent_tokens > 0 else self.encoder.num_img_tokens,
                **block_kwargs,
            )
        self.sit_loss = SILoss(
                prediction='v',
                path_type='linear', 
                encoders=[],
                weighting='lognormal'
            )

    def encode(self, x):
        
        h = self.encoder(x)
        quant = self.quant_conv(h)
        emb_loss = (torch.tensor(0.), torch.tensor(0.), torch.tensor(0.), torch.tensor(0.))
        info = None
        return quant, emb_loss, info


    def decode_train(self, quant, x):
        dec_loss, _ = self.sit_loss(self.decoder, x, {'y':quant}, zs=[])
        return dec_loss.mean()
    
    @torch.no_grad
    def decode(self, quant, num_steps=50):
        xT = torch.randn(quant.size(0), 3, self.config.image_size, self.config.image_size, device=quant.device)
        samples = euler_maruyama_sampler(
            self.decoder, 
            xT, 
            quant,
            num_steps=num_steps, 
            cfg_scale=1.0,
            guidance_low=0.,
            guidance_high=1.,
            path_type='linear',
            heun=False,
            ).to(torch.float32)
        return samples

    def forward(self, input):
        b, _, h, w = input.size()
        quant, diff, info = self.encode(input)
        
        self.quant = quant
        dec_loss = self.decode_train(quant, input)
            
        return dec_loss, diff, info


class AEJEPADiffModel(AEJEPAModel):
    def __init__(self, config: ModelArgs):
        super().__init__(config)
        self.quantize = None 
        self.num_latent_tokens = torch.tensor(config.num_latent_tokens, dtype=torch.int32)
        self.predictor = vit_predictor(
            num_patches=self.encoder.num_img_tokens,
            num_heads=self.encoder.model.num_heads,
            depth=config.predictor_depth,
            predictor_embed_dim=config.predictor_embed_dim,
            # linear predict embed dim
            embed_dim=config.codebook_embed_dim,
            num_latent_tokens=self.num_latent_tokens,
        )
        self.encoder.model.dynamic_img_size = True

        block_kwargs = {"fused_attn": False, "qk_norm": False, "flash_attn": True}
        self.decoder = SiT_models[config.decoder_model](
            image_size=config.image_size,
            patch_size=config.dec_patch_size,
            in_channels=3,
            latent_channels=config.codebook_embed_dim,
            num_latent_tokens=self.num_latent_tokens if config.num_latent_tokens > 0 else self.encoder.num_img_tokens,
            **block_kwargs,
        )
        self.sit_loss = SILoss(
            prediction='v',
            path_type='linear', 
            encoders=[],
            weighting='lognormal'
        )

    def encode(self, x, masks_enc=None, return_img_tokens=False):
        
        h = self.encoder(x, masks_enc, return_img_tokens)
        quant = self.quant_conv(h)
        emb_loss = (torch.tensor(0.), torch.tensor(0.), torch.tensor(0.), torch.tensor(0.))
        info = None
        return quant, emb_loss, info
    # def encode(self, x, masks_enc=None, return_img_tokens=False):
    #     h = self.encoder(x, masks_enc, return_img_tokens)
    #     quant = self.quant_conv(h)
    #     # 将 emb_loss 改为单个张量（避免元组）
    #     emb_loss = torch.tensor(0.0, device=x.device)
    #     info = {}  # 始终返回字典
    #     return quant, emb_loss, info

    def decode_train(self, quant, x, masks_enc=None):
        dec_loss, _ = self.sit_loss(self.decoder, x, {'y':quant, 'masks_enc':masks_enc}, zs=[])
        return dec_loss.mean()
    # def decode_train(self, quant, x, masks_enc=None):
    #     dec_loss, _ = self.sit_loss(self.decoder, x, {'y': quant, 'masks_enc': masks_enc}, zs=[])
    #     return dec_loss.mean().reshape(1)  # 确保是标量张量（非零维）
    
    @torch.no_grad()
    def decode(self, quant, num_steps=50):
        # 将 num_steps 转为张量
        num_steps_tensor = torch.tensor(num_steps, device=quant.device)
        xT = torch.randn(quant.size(0), 3, self.config.image_size, self.config.image_size, device=quant.device)
        samples = euler_maruyama_sampler(
            self.decoder, xT, quant, num_steps=num_steps_tensor  # 传入张量
        ).to(torch.float32)
        return samples
    @torch.no_grad
    def decode(self, quant, num_steps=50):
        xT = torch.randn(quant.size(0), 3, self.config.image_size, self.config.image_size, device=quant.device)
        samples = euler_maruyama_sampler(
            self.decoder, 
            xT, 
            quant,
            num_steps=num_steps, 
            cfg_scale=1.0,
            guidance_low=0.,
            guidance_high=1.,
            path_type='linear',
            heun=False,
            ).to(torch.float32)
        return samples

    def forward(self, input, masks_enc=None, masks_pred=None):
        b, _, h, w = input.size()
        quant, diff, info = self.encode(input, masks_enc)
        # print("quant, diff, info:",isinstance(quant, torch.Tensor),
        #       isinstance(diff, torch.Tensor),info)
        
        
        self.quant = quant
        if self.config.num_latent_tokens > 0:
            dec_loss = self.decode_train(quant, input)
        else:
            dec_loss = self.decode_train(quant, input, masks_enc)
        # print("dec_loss:",dec_loss)
        # jepa related
        if self.training:
            # predictor
            # print("predictor")
            pred_z = self.predictor(quant, masks_enc, masks_pred)
            # print("predictor")
            info = {'pred_z': pred_z}
            
        # assert isinstance(dec_loss, torch.Tensor)
        # assert isinstance(diff, torch.Tensor)
        # assert isinstance(info, dict) and 'pred_z' in info
            
        return dec_loss, diff, info
    # def forward(self, input, masks_enc=None, masks_pred=None):
    #     b, _, h, w = input.size()
    #     quant, diff, info = self.encode(input, masks_enc)
        
    #     # 确保 dec_loss 是标量张量
    #     if self.config.num_latent_tokens > 0:
    #         dec_loss = self.decode_train(quant, input)
    #     else:
    #         dec_loss = self.decode_train(quant, input, masks_enc)
    #     dec_loss = dec_loss.reshape(1)  # 确保形状为 [1]

    #     # 确保 info 中的值是张量
    #     if self.training:
    #         pred_z = self.predictor(quant, masks_enc, masks_pred)
    #         info = {'pred_z': pred_z}
    #     else:
    #         info = {'pred_z': torch.zeros_like(quant)}
            
    #     return dec_loss, diff, info



#################################################################################
#                              VQ Model Configs                                 #
#################################################################################
def VQ_8(**kwargs):
    return VQModel(ModelArgs(encoder_ch_mult=[1, 2, 2, 4], decoder_ch_mult=[1, 2, 2, 4], **kwargs))

def VQ_16(**kwargs):
    return VQModel(ModelArgs(encoder_ch_mult=[1, 1, 2, 2, 4], decoder_ch_mult=[1, 1, 2, 2, 4], **kwargs))

def KL_8(**kwargs):
    return KLModel(ModelArgs(encoder_ch_mult=[1, 2, 2, 4], decoder_ch_mult=[1, 2, 2, 4], **kwargs))

def KL_16(**kwargs):
    return KLModel(ModelArgs(encoder_ch_mult=[1, 1, 2, 2, 4], decoder_ch_mult=[1, 1, 2, 2, 4], **kwargs))

def AE_16(**kwargs):
    return AEModel(ModelArgs(encoder_ch_mult=[1, 1, 2, 2, 4], decoder_ch_mult=[1, 1, 2, 2, 4], **kwargs))

def AEJEPA_16(**kwargs):
    return AEJEPAModel(ModelArgs(encoder_ch_mult=[1, 1, 2, 2, 4], decoder_ch_mult=[1, 1, 2, 2, 4], **kwargs))

def AEDiff_16(**kwargs):
    return AEDiffModel(ModelArgs(encoder_ch_mult=[1, 1, 2, 2, 4], decoder_ch_mult=[1, 1, 2, 2, 4], **kwargs))

def AEJEPADiff_16(**kwargs):
    return AEJEPADiffModel(ModelArgs(encoder_ch_mult=[1, 1, 2, 2, 4], decoder_ch_mult=[1, 1, 2, 2, 4], **kwargs))

def SoftVQ(**kwargs):
    return SoftVQModel(ModelArgs(encoder_ch_mult=[1, 1, 2, 2, 4], decoder_ch_mult=[1, 1, 2, 2, 4], **kwargs))


VQ_models = {
    'AE-16': AE_16,
    'AE-Diff-16': AEDiff_16,
    'AE-JEPA-16': AEJEPA_16,
    'AE-JEPA-Diff-16': AEJEPADiff_16,
    'VQ-16': VQ_16, 'VQ-8': VQ_8,
    'KL-16': KL_16, 'KL-8': KL_8,
    'SoftVQ': SoftVQ,
    }


if __name__ == '__main__':
    
    # model = VQ_16(codebook_embed_dim=16, enc_type='vit', dec_type='vit', encoder_model='vit_base_patch14_dinov2.lvd142m', decoder_model='vit_base_patch14_dinov2.lvd142m', repa=True, repa_model='vit_base_patch14_dinov2.lvd142m', repa_align='avg_1d_shuffle', enc_img_res=True, enc_img_align='avg_1d', dec_img_res=True)    
    # model = SoftVQ_16(codebook_embed_dim=16, enc_type='vit', dec_type='vit', encoder_model='vit_base_patch14_dinov2.lvd142m', decoder_model='vit_base_patch14_dinov2.lvd142m', num_codebooks=4, codebook_size=16384, topk=16)
    # model = AE_16(codebook_embed_dim=16, enc_type='vit', dec_type='vit', encoder_model='vit_base_patch14_dinov2.lvd142m', decoder_model='vit_base_patch14_dinov2.lvd142m', num_codebooks=4, codebook_size=16384)
    model = AEJEPADiff_16(codebook_embed_dim=16, enc_type='vit', dec_type='sit', encoder_model='vit_base_patch14_dinov2.lvd142m', decoder_model='SiT-B', num_latent_tokens=16)
    model.train()
    # model = KL_16(codebook_embed_dim=16, enc_type='vit', dec_type='vit', encoder_model='vit_base_patch14_dinov2.lvd142m', decoder_model='vit_base_patch14_dinov2.lvd142m')
    # model = GMM_16(codebook_embed_dim=16, enc_type='vit', dec_type='vit', encoder_model='vit_base_patch14_dinov2.lvd142m', decoder_model='vit_base_patch14_dinov2.lvd142m')
    x = torch.randn(1, 3, 256, 256)
    y, _, info = model(x, [], [])
