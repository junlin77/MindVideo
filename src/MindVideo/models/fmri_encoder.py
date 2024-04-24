from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models import ModelMixin
from diffusers.utils.import_utils import is_xformers_available
from .. import get_1d_sincos_pos_embed, interpolate_pos_embed
import torch
import torch.nn as nn
import numpy as np
from timm.models.layers import Mlp, DropPath
import torch.nn.functional as F
from typing import Dict, Optional, Tuple,Callable
import os, sys
import json
from einops import rearrange

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(project_root)
import utils.utils as ut

if is_xformers_available():
    import xformers
    import xformers.ops
else:
    xformers = None

class PatchEmbed1D(nn.Module):
    """ 1 Dimensional version of data (fmri voxels) to Patch Embedding
    """
    def __init__(self, num_voxels=224, patch_size=16, in_chans=1, embed_dim=768):
        super().__init__()
        num_patches = num_voxels // patch_size
        self.patch_shape = patch_size
        self.num_voxels = num_voxels
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv1d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x, **kwargs):
        B, C, V = x.shape # batch, channel, voxels
        # assert V == self.num_voxels, \
        #     f"Input fmri length ({V}) doesn't match model ({self.num_voxels})."
        x = self.proj(x).transpose(1, 2).contiguous() # put embed_dim at the last dimension
        return x
    
class ProjectionHead(nn.Module):
    def __init__(self, 
                 in_channel=1, 
                 out_channel=1, 
                 in_dim=1024, 
                 out_dim=128, 
                 dropout=.0, 
                 use_norm=False,
                 use_norm_in=False):
        super().__init__()
        self.projection = nn.Linear(in_dim, out_dim, bias=False)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(out_dim, out_dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(out_dim) if use_norm else nn.Identity() 
        self.layer_norm_in = nn.LayerNorm(in_dim) if use_norm_in else nn.Identity() 
        self.channel_mapper = nn.Linear(in_channel, out_channel, bias=False)
        self.in_channel = in_channel
        self.out_channel = out_channel
        # print(f'ProjectionHead: in_channel={in_channel}, out_channel={out_channel}, in_dim={in_dim}, out_dim={out_dim}, dropout={dropout}, use_norm={use_norm}')
    def forward(self, x):  
        x = self.layer_norm_in(x)
        x = x.transpose(1, 2)
        x = self.channel_mapper(x) 
        x = x.transpose(1, 2)
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + projected
        x = self.layer_norm(x)
        return x # n, out_channel, out_dim


class Attention(nn.Module):
    _use_memory_efficient_attention_xformers = False
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.drop_rate = attn_drop
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def set_use_memory_efficient_attention_xformers(self, use_memory_efficient_attention_xformers: bool, 
                                                     attention_op: Optional[Callable] = None):
        if not is_xformers_available():
            print("Here is how to install it")
            raise ModuleNotFoundError(
                "Refer to https://github.com/facebookresearch/xformers for more information on how to install"
                " xformers",
                name="xformers",
            )
        elif not torch.cuda.is_available():
            raise ValueError(
                "torch.cuda.is_available() should be True but is False. xformers' memory efficient attention is only"
                " available for GPU "
            )
        else:
            try:
                # Make sure we can run the memory efficient attention
                _ = xformers.ops.memory_efficient_attention(
                    torch.randn((1, 2, 40), device="cuda"),
                    torch.randn((1, 2, 40), device="cuda"),
                    torch.randn((1, 2, 40), device="cuda"),
                )
            except Exception as e:
                raise e
        self._use_memory_efficient_attention_xformers = use_memory_efficient_attention_xformers
        self.attention_op = attention_op
        # print('fmri_encoder: use xformers attention')

    def batch_to_head_dim(self, tensor):
        head_size = self.num_heads
        batch_size, seq_len, dim = tensor.shape
        tensor = tensor.reshape(batch_size // head_size, head_size, seq_len, dim)
        tensor = tensor.permute(0, 2, 1, 3).reshape(batch_size // head_size, seq_len, dim * head_size)
        return tensor
    def head_to_batch_dim(self, tensor):
        head_size = self.num_heads
        batch_size, seq_len, dim = tensor.shape
        tensor = tensor.reshape(batch_size, seq_len, head_size, dim // head_size)
        tensor = tensor.permute(0, 2, 1, 3).reshape(batch_size * head_size, seq_len, dim // head_size)
        return tensor
     
    def forward(self, x, return_attn=False):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)
        # q: B, num_heads, N, C // num_heads
        # k: B, num_heads, N, C // num_heads
        # v: B, num_heads, N, C // num_heads
        if return_attn:
            assert not self._use_memory_efficient_attention_xformers, 'return_attn is not supported with xformers'
            assert not self.training, 'return_attn is not supported in training mode'
        if self._use_memory_efficient_attention_xformers:
            q = q.reshape(B * self.num_heads, N, C // self.num_heads)
            k = k.reshape(B * self.num_heads, N, C // self.num_heads)
            v = v.reshape(B * self.num_heads, N, C // self.num_heads)
            x = xformers.ops.memory_efficient_attention(
                q, k, v, attn_bias=None, op=self.attention_op, scale=self.scale, p=self.drop_rate
            )
            x = x.to(q.dtype)
            x = self.batch_to_head_dim(x) # B, N, C
        else:
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            if return_attn:
                return attn
            
            x = (attn @ v).transpose(1, 2).reshape(B, N, C)
            

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.gradient_checkpointing = False
    
    def forward(self, x, **kwargs):
        if self.training and self.gradient_checkpointing:

            def create_custom_forward(module):
                def custom_forward(*inputs):
                    return module(*inputs)

                return custom_forward
            x = x + self.drop_path(torch.utils.checkpoint.checkpoint(create_custom_forward(self.attn), self.norm1(x)))
            x = x + self.drop_path(torch.utils.checkpoint.checkpoint(create_custom_forward(self.mlp), self.norm2(x)))
        else:    
            x = x + self.drop_path(self.attn(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class BlockTemp(nn.Module):
    
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.attn_temp = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.norm_temp = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.gradient_checkpointing = False
    
    def forward(self, x, window_size=None, return_attn=False, **kwargs):
        if return_attn:
            assert not self.training, 'return_attn is not supported in training mode'

        if self.training and self.gradient_checkpointing:

            def create_custom_forward(module):
                def custom_forward(*inputs):
                    return module(*inputs)

                return custom_forward
            x = x + self.drop_path(torch.utils.checkpoint.checkpoint(create_custom_forward(self.attn), self.norm1(x)))
            # Temporal-Attention
            d = x.shape[1]
            x = rearrange(x, "(b f) d c -> (b d) f c", f=window_size)
            x = x + self.drop_path(torch.utils.checkpoint.checkpoint(create_custom_forward(self.attn_temp), self.norm_temp(x)))
            x = rearrange(x, "(b d) f c -> (b f) d c", d=d)
            x = x + self.drop_path(torch.utils.checkpoint.checkpoint(create_custom_forward(self.mlp), self.norm2(x)))
        else:    
            if return_attn:
                spatial_attn = self.attn(self.norm1(x), return_attn=return_attn)
                x = x + self.drop_path(self.attn(self.norm1(x)))
                d = x.shape[1]
                x = rearrange(x, "(b f) d c -> (b d) f c", f=window_size)
                temporal_attn = self.attn_temp(self.norm_temp(x), return_attn=return_attn)
                return spatial_attn, temporal_attn
            
            x = x + self.drop_path(self.attn(self.norm1(x)))
            # Temporal-Attention
            d = x.shape[1]
            x = rearrange(x, "(b f) d c -> (b d) f c", f=window_size)
            x = x + self.drop_path(self.attn_temp(self.norm_temp(x)))
            x = rearrange(x, "(b d) f c -> (b f) d c", d=d)
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class MAEforFMRI(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self, num_voxels=224, patch_size=16, embed_dim=1024, in_chans=1,
                 depth=24, num_heads=16, decoder_embed_dim=512, 
                 decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, focus_range=None, focus_rate=None, img_recon_weight=1.0, 
                 use_nature_img_loss=False):
        super().__init__()

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.patch_embed = PatchEmbed1D(num_voxels, patch_size, in_chans, embed_dim)

        num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size * in_chans, bias=True) # encoder to decoder
        # --------------------------------------------------------------------------

        # nature image decoder specifics
        if use_nature_img_loss:
            self.nature_img_decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

            self.nature_img_mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

            self.nature_img_decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding

            self.nature_img_decoder_blocks = nn.ModuleList([
                Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
                for i in range(2)])

            self.nature_img_decoder_norm = norm_layer(decoder_embed_dim)
            self.nature_img_decoder_pred = nn.Sequential(
                nn.Conv1d(num_patches, 512, kernel_size=1, stride=1, bias=True),
                nn.Linear(decoder_embed_dim, 28*28, bias=True)
            )
            # --------------------------------------------------------------------------

        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.focus_range = focus_range
        self.focus_rate = focus_rate
        self.img_recon_weight = img_recon_weight
        self.use_nature_img_loss = use_nature_img_loss
   
        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = ut.get_1d_sincos_pos_embed(self.pos_embed.shape[-1], self.patch_embed.num_patches, cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = ut.get_1d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], self.patch_embed.num_patches, cls_token=True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        if self.use_nature_img_loss:
            nature_img_decoder_pos_embed = ut.get_1d_sincos_pos_embed(self.nature_img_decoder_pos_embed.shape[-1], self.patch_embed.num_patches, cls_token=True)
            self.nature_img_decoder_pos_embed.data.copy_(torch.from_numpy(nature_img_decoder_pos_embed).float().unsqueeze(0))
            torch.nn.init.normal_(self.nature_img_mask_token, std=.02)

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            torch.nn.init.normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    
    def patchify(self, imgs):
        """
        imgs: (N, 1, num_voxels)
        x: (N, L, patch_size)
        """
        p = self.patch_embed.patch_size
        assert imgs.ndim == 3 and imgs.shape[2] % p == 0

        h = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], h, p))
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size)
        imgs: (N, 1, num_voxels)
        """
        p = self.patch_embed.patch_size
        h = x.shape[1]
        
        imgs = x.reshape(shape=(x.shape[0], 1, h * p))
        return imgs

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        if self.focus_range is not None:
            len_mask = L - len_keep
            weights = [1-self.focus_rate] * L
            weights[self.focus_range[0] // self.patch_size : self.focus_range[1] // self.patch_size
                        ] = [self.focus_rate] * (self.focus_range[1] // self.patch_size - self.focus_range[0] // self.patch_size)
            weights = torch.tensor(weights).repeat(N, 1).to(x.device)
            ids_mask = torch.multinomial(weights, len_mask, replacement=False)
            
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        if self.focus_range is not None:
            for i in range(N):
                noise[i, ids_mask[i,:]] = 1.1  # set mask portion to 1.1 

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward_encoder(self, x, mask_ratio):
        # embed patches
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        # masking: length -> length * mask_ratio
        x, mask, ids_restore = self.random_masking(x, mask_ratio)

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x, mask, ids_restore

    def forward_decoder(self, x, ids_restore):
        # embed tokens
        x = self.decoder_embed(x)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        # add pos embed
        x = x + self.decoder_pos_embed

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        # remove cls token
        x = x[:, 1:, :]

        return x

    def forward_nature_img_decoder(self, x, ids_restore):
        # embed tokens
        x = self.nature_img_decoder_embed(x)

        # append mask tokens to sequence
        mask_tokens = self.nature_img_mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        # add pos embed
        x = x + self.nature_img_decoder_pos_embed

        # apply Transformer blocks
        for blk in self.nature_img_decoder_blocks:
            x = blk(x)
        x = self.nature_img_decoder_norm(x)
        # remove cls token
        x = x[:, 1:, :]
        # predictor projection
        # x = x.mean(dim=1, keepdim=True)
        x = self.nature_img_decoder_pred(x)
        x = x.view(x.shape[0], 512, 28, 28)

        return x # n, 512, 28, 28
        
    def forward_nature_img_loss(self, inputs, reconstructions):
        loss = ((torch.tanh(inputs) - torch.tanh(reconstructions))**2).mean()
        if torch.isnan(reconstructions).sum():
            print('nan in reconstructions')
        if torch.isnan(inputs).sum():
            print('nan in inputs')
    
        return loss   

    def forward_loss(self, imgs, pred, mask):
        """
        imgs: [N, 1, num_voxels]
        pred: [N, L, p]
        mask: [N, L], 0 is keep, 1 is remove, 
        """
        target = self.patchify(imgs)

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch
        
        loss = (loss * mask).sum() / mask.sum()  if mask.sum() != 0 else (loss * mask).sum() # mean loss on removed patches
        return loss

    def forward(self, imgs, img_features=None, valid_idx=None, mask_ratio=0.75):
        latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio)
        pred = self.forward_decoder(latent, ids_restore)  # [N, L, p]
        loss = self.forward_loss(imgs, pred, mask)

        if self.use_nature_img_loss and img_features is not None:
            # valid_idx = torch.nonzero(nature_image.sum(dim=(1,2,3)) != 0).squeeze(1)
            if len(valid_idx) != 0:
                nature_image_recon = self.forward_nature_img_decoder(latent[valid_idx], ids_restore[valid_idx])
                loss_nature_image_recon = self.forward_nature_img_loss(img_features, nature_image_recon)
                if torch.isnan(loss_nature_image_recon).sum():
                    print(loss_nature_image_recon)
                    print("loss_nature_image_recon is nan")
                    
                loss = loss + self.img_recon_weight*loss_nature_image_recon

        return loss, pred, mask

class fMRIEncoder(ModelMixin, ConfigMixin):
    config_name = 'config.json'
    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(self, 
                 num_voxels: int=224, 
                 patch_size: int=16, 
                 embed_dim: int=1024, 
                 in_chans: int=1,
                 depth: int=24, 
                 num_heads: int=16, 
                 mlp_ratio: float=4.,
                 # project head setting
                 out_channel: int=1,
                 out_dim: int=768,
                 use_norm: bool=False,
                 dropout: float=.0,
                 window_size: int=1,
                 use_temp_attn: bool=False,
                 logit_scale_init_value: float=1.0,
                 **kwargs):
        
        super().__init__()

        self.patch_embed = PatchEmbed1D(num_voxels, patch_size, in_chans, embed_dim)

        num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)  # fixed sin-cos embedding

        norm_layer = nn.LayerNorm
        vit_block = BlockTemp if use_temp_attn else Block
        self.blocks = nn.ModuleList([
            vit_block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer, drop_path=dropout)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
    
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.proj_head = ProjectionHead(in_channel=num_patches * window_size, out_channel=out_channel, in_dim=embed_dim, out_dim=out_dim, use_norm=use_norm, dropout=dropout)
        self.logit_scale = nn.Parameter(torch.ones([]) * logit_scale_init_value)

        self.embed_dim = embed_dim
        self.window_size = window_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.initialize_weights()
        self.gradient_checkpointing = False

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_1d_sincos_pos_embed(self.pos_embed.shape[-1], self.patch_embed.num_patches, cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)
        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            torch.nn.init.normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    @torch.no_grad()
    def forward_attn(self, x, layer):
        # embed patches
        assert self.window_size == x.shape[1], f'window_size {self.window_size} != x.shape[1] {x.shape[1]}'
        window_size = x.shape[1]
        x = rearrange(x, 'b c n -> (b c) 1 n')
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]
        # apply Transformer blocks
        for i, blk in enumerate(self.blocks):
            if i != layer:
                x = blk(x, window_size=window_size)
            else:
                spatial_attn, temp_attn = blk(x, window_size=window_size, return_attn=True)
                break
        # spatial_attn = rearrange(spatial_attn, '(b c) n d -> b c n d', c=window_size)
        return (spatial_attn / spatial_attn.norm(p=2, dim=-1, keepdim=True), 
                temp_attn / temp_attn.norm(p=2, dim=-1, keepdim=True))
        

    def forward_encoder(self, x):
        # embed patches
        assert self.window_size == x.shape[1], f'window_size {self.window_size} != x.shape[1] {x.shape[1]}'
        window_size = x.shape[1]
        x = rearrange(x, 'b c n -> (b c) 1 n')
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]
        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x, window_size=window_size)
        x = self.norm(x) # output: N * window_size, n_seq, embed_dim
        x = rearrange(x, '(b c) n d -> b (c n) d', c=window_size)

        if self.gradient_checkpointing:
            def create_custom_forward(module):
                def custom_forward(*inputs):
                    return module(*inputs)
                return custom_forward
            x = torch.utils.checkpoint.checkpoint(create_custom_forward(self.proj_head), x)
        else:
            x = self.proj_head(x)
        return x  

    def forward(self, imgs):
        if imgs.ndim == 1:
            imgs = imgs[None, None, :]
        if imgs.ndim == 2:
            imgs = imgs[:, None, :]
        # expected input shape: N, 1, num_voxels
        latent = self.forward_encoder(imgs) # N, n_seq, embed_dim
        return latent # N, n_seq, embed_dim
    
    def load_checkpoint(self, state_dict):
        state_dict = {k: v for k, v in state_dict.items() if ('mask_token' not in k)}
        interpolate_pos_embed(self, state_dict)
            
        m, u = self.load_state_dict(state_dict, strict=False)
        print('missing keys:', u)
        print('unexpected keys:', m)
        return 
    
    @classmethod
    def from_pretrained(cls, pretrained_model_path, num_voxels, subfolder=None):
        if subfolder is not None:
            pretrained_model_path = os.path.join(pretrained_model_path, subfolder)
        model = cls.from_config_path(pretrained_model_path, num_voxels)
        ckp_file = os.path.join(pretrained_model_path, 'model.pth' 
                                if 'model.pth' in os.listdir(pretrained_model_path) else 'diffusion_pytorch_model.bin')
        if not os.path.isfile(ckp_file):
            raise RuntimeError(f"{ckp_file} does not exist")
        ckp = torch.load(ckp_file, map_location='cpu')
        ckp = ckp['model'] if 'model' in ckp.keys() else ckp
        model.load_checkpoint(ckp)
        return model
    
    @classmethod
    def from_config_path(cls, config_path, num_voxels, subfolder=None):
        if subfolder is not None:
            config_path = os.path.join(config_path, subfolder)
        config_file = os.path.join(config_path, 'config.json')
        if not os.path.isfile(config_file):
            raise RuntimeError(f"{config_file} does not exist")
        with open(config_file, "r") as f:
            config = json.load(f)
        config['num_voxels'] = num_voxels
        model = cls.from_config(config)
        return model
    
    def _set_gradient_checkpointing(self, module, value=False):
        self.gradient_checkpointing = True
        if isinstance(module, Block):
            # print('fmri_encoder: enable gradient_checkpointing')
            module.gradient_checkpointing = value