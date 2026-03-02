
# coding: utf-8
"""
modeling_fineEMenc.py: Encoder for fine-tuning (PredANN++)

Encoder structure shared with:
- ms20_modeling_preEMenc.py
- ms40_modeling_preEMenc.py

License:
- This file is released under CC-BY-SA 4.0 as part of this repository (see LICENSE).
- IMPORTANT THIRD-PARTY NOTICE:
  Portions of this file are adapted from LaBraM (MIT License), specifically:
    - TemporalConv
    - drop_path
    - DropPath
    - Mlp
    - Attention
    - Block
  We add explicit attribution below and keep the MIT license notice.
"""

import math
from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F

# External libraries: einops, timm
from einops import rearrange
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_

# ============================================================================
# Third-Party Code Notice (LaBraM, MIT License)
# ----------------------------------------------------------------------------
# The following components are adapted from LaBraM's modeling_finetune.py:
#   - TemporalConv
#   - drop_path
#   - DropPath
#   - Mlp
#   - Attention
#   - Block
#
# LaBraM is stated as MIT-licensed by the upstream project (as provided by the authors).
# This repository provides explicit attribution for transparency and compliance.
#
# MIT License (summary):
# - Permission is hereby granted, free of charge, to use/copy/modify/merge/publish/distribute.
# - The above copyright notice and this permission notice shall be included in copies.
# ============================================================================

class TemporalConv(nn.Module):
    """EEG to patch embedding via temporal convolutions."""
    def __init__(self, in_chans: int = 1, out_chans: int = 8):
        super().__init__()
        self.conv1 = nn.Conv2d(in_chans, out_chans, kernel_size=(1, 15), stride=(1, 8), padding=(0, 7))
        self.norm1 = nn.GroupNorm(4, out_chans)
        self.conv2 = nn.Conv2d(out_chans, out_chans, kernel_size=(1, 3), padding=(0, 1))
        self.norm2 = nn.GroupNorm(4, out_chans)
        self.conv3 = nn.Conv2d(out_chans, out_chans, kernel_size=(1, 3), padding=(0, 1))
        self.norm3 = nn.GroupNorm(4, out_chans)
        self.gelu = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = rearrange(x, 'B C S T -> B (C S) T').unsqueeze(1)  
        x = self.gelu(self.norm1(self.conv1(x)))
        x = self.gelu(self.norm2(self.conv2(x)))
        x = self.gelu(self.norm3(self.conv3(x)))
        x = rearrange(x, 'B C N T -> B N (T C)')             
        return x

class PatchEEG(nn.Module):
    def __init__(self, in_chans: int = 1, out_chans: int = 8):
        super().__init__()
        self.temporal_conv = TemporalConv(in_chans=in_chans, out_chans=out_chans)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.temporal_conv(x)
        return x


def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path)."""
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,)*(x.ndim-1)
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    output = x / keep_prob * random_tensor
    return output

class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob
    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)
    def extra_repr(self) -> str:
        return f"p={self.drop_prob}"

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
    def forward(self, x):
        # x: [B, N, C]
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Attention(nn.Module):
    def __init__(self,
                 dim,
                 num_heads=8,
                 qkv_bias=False,
                 qk_norm=None,
                 qk_scale=None,
                 attn_drop=0.,
                 proj_drop=0.,
                 window_size=None,
                 attn_head_dim=None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.qkv = nn.Linear(dim, all_head_dim * 3, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(all_head_dim))
            self.v_bias = nn.Parameter(torch.zeros(all_head_dim))
        else:
            self.q_bias = None
            self.v_bias = None

        self.q_norm = qk_norm(head_dim) if qk_norm else None
        self.k_norm = qk_norm(head_dim) if qk_norm else None

        self.window_size = window_size
        self.relative_position_bias_table = None
        self.relative_position_index = None

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, rel_pos_bias=None,
                return_attention=False, return_qkv=False):
        B, N, C = x.shape
        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat([self.q_bias,
                                  torch.zeros_like(self.v_bias),
                                  self.v_bias], dim=0)

        qkv = F.linear(x, weight=self.qkv.weight, bias=qkv_bias)
        # => [B, N, 3, num_heads, head_dim], then permute
        qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2,0,3,1,4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        if self.q_norm is not None:
            q = self.q_norm(q)
        if self.k_norm is not None:
            k = self.k_norm(k)

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        if rel_pos_bias is not None:
            attn = attn + rel_pos_bias

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        if return_attention:
            return attn

        x_out = (attn @ v).transpose(1,2).reshape(B, N, -1)
        x_out = self.proj(x_out)
        x_out = self.proj_drop(x_out)

        if return_qkv:
            return x_out, qkv
        return x_out

class Block(nn.Module):
    def __init__(self,
                 dim,
                 num_heads,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_norm=None,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 init_values=0.1,
                 act_layer=nn.GELU,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 window_size=None,
                 attn_head_dim=None):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            window_size=window_size,
            attn_head_dim=attn_head_dim)
        self.drop_path = DropPath(drop_path) if drop_path>0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        hidden_dim = int(dim*mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=hidden_dim,
                       out_features=dim, act_layer=act_layer, drop=drop)

        if init_values and init_values>0:
            self.gamma_1 = nn.Parameter(init_values*torch.ones((dim)), requires_grad=True)
            self.gamma_2 = nn.Parameter(init_values*torch.ones((dim)), requires_grad=True)
        else:
            self.gamma_1, self.gamma_2 = None, None

    def forward(self, x, rel_pos_bias=None,
                return_attention=False, return_qkv=False):
        if return_attention:
            attn = self.attn(self.norm1(x), rel_pos_bias=rel_pos_bias, return_attention=True)
            return attn
        if return_qkv:
            y, qkv = self.attn(self.norm1(x), rel_pos_bias=rel_pos_bias, return_qkv=True)
            if self.gamma_1 is not None:
                x = x + self.drop_path(self.gamma_1*y)
                x = x + self.drop_path(self.gamma_2*self.mlp(self.norm2(x)))
            else:
                x = x + self.drop_path(y)
                x = x + self.drop_path(self.mlp(self.norm2(x)))
            return x, qkv

        if self.gamma_1 is not None:
            x = x + self.drop_path(self.gamma_1*self.attn(self.norm1(x), rel_pos_bias=rel_pos_bias))
            x = x + self.drop_path(self.gamma_2*self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.attn(self.norm1(x), rel_pos_bias=rel_pos_bias))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

egi_128_elecNames = [f"E{i}" for i in range(1,129)]

def get_eeg_channel_index(ch_names: list):
    """
    ch_names: List of electrode names used by the user.
    Returns: Index list from egi_128_elecNames
    """
    idx_list = []
    for c in ch_names:
        if c in egi_128_elecNames:
            idx_list.append( egi_128_elecNames.index(c) )
        else:
            # unknown => skip or fallback
            pass
    return idx_list


class Comp1FineEMEncoder(nn.Module):
    """Encoder for fine-tuning: EEG(384) → transformer blocks."""
    def __init__(self,
                 embed_dim: int = 512,
                 num_heads: int = 8,
                 mlp_ratio: float = 4.0,
                 qkv_bias: bool = False,
                 qk_norm=partial(nn.LayerNorm, eps=1e-6),
                 qk_scale: float = None,
                 drop_rate: float = 0.1,
                 attn_drop_rate: float = 0.0,
                 drop_path_rate: float = 0.0,
                 init_values: float = 0.1,
                 depth_enc: int = 2, # Number of encoder layers
                 depth_dec: int = 2, # Decoder is not used in FullScratch/Finetune
                 window_size: int = None,
                 attn_head_dim: int = None,
                 music_use_common_time_embed: bool = True,
                 use_cls_token: bool = True,
                 **kwargs):
        super().__init__()
        self.patch_eeg  = PatchEEG()
        self.eeg_linear = nn.Linear(128, embed_dim)

        self.use_cls_token = use_cls_token
    
        if self.use_cls_token:
            self.cls_token  = nn.Parameter(torch.zeros(1, 1, embed_dim))
        else:
            self.cls_token = None

        self.eeg_ch_emb = nn.Parameter(torch.zeros(128, embed_dim))
        self.sec_emb    = nn.Parameter(torch.zeros(3,   embed_dim))



        # --- Encoder Blocks ---
        dpr_enc = [x.item() for x in torch.linspace(0, drop_path_rate, depth_enc)]
        self.encoder = nn.Sequential(*[
            Block(embed_dim, num_heads,
                  mlp_ratio, qkv_bias,
                  qk_norm, qk_scale,
                  drop_rate, attn_drop_rate, dpr_enc[i],
                  init_values,
                  norm_layer=partial(nn.LayerNorm, eps=1e-6),
                  window_size=window_size,
                  attn_head_dim=attn_head_dim)
            for i in range(depth_enc)
        ])

        # --- Decoder Blocks ---
        #dpr_dec = [x.item() for x in torch.linspace(0, drop_path_rate, depth_dec)]
        #self.decoder = nn.Sequential(*[
        #    Block(embed_dim, num_heads,
        #          mlp_ratio, qkv_bias,
        #          qk_norm, qk_scale,
        #          drop_rate, attn_drop_rate, dpr_dec[i],
        #          init_values,
        #          norm_layer=partial(nn.LayerNorm, eps=1e-6),
        #          window_size=window_size,
        #          attn_head_dim=attn_head_dim)
        #    for i in range(depth_dec)
        #])
        #self.decoder_mask_norm = nn.LayerNorm(embed_dim)

        # --- Projection ---
        #self.proj_out = nn.Linear(embed_dim, 128) #changed from 1024

        # --- Init ---
#ここから{CLS分岐}>>
        if self.use_cls_token:
            trunc_normal_(self.cls_token, std=0.02)
#<<ここまで{CLS分岐}
        trunc_normal_(self.eeg_ch_emb, std=0.02)
        trunc_normal_(self.sec_emb,    std=0.02)
        #trunc_normal_(self.time20_emb, std=0.02)
        #trunc_normal_(self.mask_token, std=0.02)
        trunc_normal_(self.eeg_linear.weight, std=0.02)
        nn.init.zeros_(self.eeg_linear.bias)

    @torch.jit.ignore
    def no_weight_decay(self):
            """Exclude embeddings and LayerScale parameters from weight decay."""
            names = {
               # "mask_token",
                "eeg_ch_emb",
                "sec_emb",
               # "time20_emb",
            }
            # LayerScale γ (for each Block in encoder/decoder)
            for i in range(len(self.encoder)):
                names.update({f"encoder.{i}.gamma_1", f"encoder.{i}.gamma_2"})
            #for i in range(len(self.decoder)):
            #    names.update({f"decoder.{i}.gamma_1", f"decoder.{i}.gamma_2"})

            #names.update({
                # "decoder_mask_norm.weight",
                #"decoder_mask_norm.bias",
            #})
            if self.use_cls_token:
                names.update({"cls_token"})
            return names



    # forward
    def forward(self, eeg_raw: torch.Tensor, ch_name_list=None):
        B = eeg_raw.size(0)

        z = self.patch_eeg(eeg_raw)
        z = self.eeg_linear(z)

        z = z.view(B, 128, 3, -1)   
        idx = list(range(128))
        z = z + self.eeg_ch_emb[idx].unsqueeze(1)
        z = z + self.sec_emb.unsqueeze(0).unsqueeze(0)
        z = z.view(B, 384, -1)

        if self.use_cls_token:
            cls_tok = self.cls_token.expand(B, -1, -1)
            s_enc   = torch.cat([cls_tok, z], dim=1)
            h_enc   = self.encoder(s_enc)
            h_repr  = h_enc[:, 0, :]
        else:
            s_enc   = z
            h_enc   = self.encoder(s_enc)
            h_repr  = h_enc.mean(dim=1)

        return h_repr
    
    def forward_return_s_enc(self, eeg_raw: torch.Tensor):
        B = eeg_raw.size(0)

        z = self.patch_eeg(eeg_raw)
        z = self.eeg_linear(z)

        z = z.view(B, 128, 3, -1)
        idx = list(range(128))
        z = z + self.eeg_ch_emb[idx].unsqueeze(1)
        z = z + self.sec_emb.unsqueeze(0).unsqueeze(0)
        z = z.view(B, 384, -1)

        if self.use_cls_token:
            cls_tok = self.cls_token.expand(B, -1, -1)
            s_enc   = torch.cat([cls_tok, z], dim=1)
        else:
            s_enc   = z

        return s_enc


# timm registry
from typing import Any


@register_model
def comp1_fineEEGenc_2layer_512(
    *,
    pretrained: bool = False,
    init_ckpt: str = "",
    depth_enc: int = 2,
    use_cls_token: bool = True,
    **kwargs: Any,
):
    
    for k in ["pretrained_cfg","pretrained_cfg_overlay", "features_only","cache_dir",  "scriptable"]:
        kwargs.pop(k, None)

    model = Comp1FineEMEncoder(depth_enc=2, use_cls_token=use_cls_token, **kwargs)
    if pretrained and init_ckpt:
        checkpoint = torch.load(init_ckpt, map_location="cpu")
        if "emenc_state_dict" in checkpoint:
            state_dict = checkpoint["emenc_state_dict"]
        elif "module_state_dict" in checkpoint:
            state_dict = {k.replace("emenc.", ""): v
                          for k, v in checkpoint["module_state_dict"].items()
                          if k.startswith("emenc.")}
        else:
            state_dict = checkpoint

        new_sd = {}
        for k, v in state_dict.items():
            if k.startswith("decoder") or k.startswith("proj_out") or k == "mask_token" or k.startswith("time40_emb"):
                print(f"[DROP] {k} skipped (not used in Finetune model)")
                continue

            if (not model.use_cls_token) and k == "cls_token":
                print(f"[DROP] {k} skipped (CLS disabled)")
                continue

            new_sd[k] = v
        missing, unexpected = model.load_state_dict(new_sd, strict=True)
        assert not missing and not unexpected, "strict=True failed"

    return model
