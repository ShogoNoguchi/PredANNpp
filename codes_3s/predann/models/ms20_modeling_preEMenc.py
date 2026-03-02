#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Pre-training encoder-decoder model for MuQ prediction at 20ms intervals."""

import math
from functools import partial

import torch
import torch.nn as nn

# External library: timm
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_

from predann.models.modeling_fineEMenc import (
    TemporalConv,
    PatchEEG,
    drop_path,
    DropPath,
    Mlp,
    Attention,
    Block,
    egi_128_elecNames,
    get_eeg_channel_index,
)

# Encoder-Decoder Pre-training Model
class Comp1EDPretrain(nn.Module):
    """
    Encoder F : EEG(384)  → 2×Block
    Decoder G : [F(EEG) ‖ mask×150] → 2×Block
    Output: (B,150,128)
    """

    def __init__(
        self,
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
        depth_enc: int = 2,
        depth_dec: int = 2,
        window_size: int = None,
        attn_head_dim: int = None,
        music_use_common_time_embed: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.patch_eeg = PatchEEG()
        self.eeg_linear = nn.Linear(128, embed_dim)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # --- Embeddings ---
        self.eeg_ch_emb = nn.Parameter(torch.zeros(128, embed_dim))
        self.sec_emb = nn.Parameter(torch.zeros(3, embed_dim))
        self.time20_emb = nn.Parameter(torch.zeros(150, embed_dim))
        self.mask_token = nn.Parameter(torch.zeros(1, embed_dim))
        self.music_feat_proj = nn.Linear(1, embed_dim)

        # --- Encoder Blocks ---
        dpr_enc = [x.item() for x in torch.linspace(0, drop_path_rate, depth_enc)]
        self.encoder = nn.Sequential(
            *[
                Block(
                    embed_dim,
                    num_heads,
                    mlp_ratio,
                    qkv_bias,
                    qk_norm,
                    qk_scale,
                    drop_rate,
                    attn_drop_rate,
                    dpr_enc[i],
                    init_values,
                    norm_layer=partial(nn.LayerNorm, eps=1e-6),
                    window_size=window_size,
                    attn_head_dim=attn_head_dim,
                )
                for i in range(depth_enc)
            ]
        )

        # --- Decoder Blocks ---
        dpr_dec = [x.item() for x in torch.linspace(0, drop_path_rate, depth_dec)]
        self.decoder = nn.Sequential(
            *[
                Block(
                    embed_dim,
                    num_heads,
                    mlp_ratio,
                    qkv_bias,
                    qk_norm,
                    qk_scale,
                    drop_rate,
                    attn_drop_rate,
                    dpr_dec[i],
                    init_values,
                    norm_layer=partial(nn.LayerNorm, eps=1e-6),
                    window_size=window_size,
                    attn_head_dim=attn_head_dim,
                )
                for i in range(depth_dec)
            ]
        )

        # --- Projection ---
        self.proj_out = nn.Linear(embed_dim, 128)

        # --- Init ---
        trunc_normal_(self.cls_token, std=0.02)
        trunc_normal_(self.eeg_ch_emb, std=0.02)
        trunc_normal_(self.sec_emb, std=0.02)
        trunc_normal_(self.time20_emb, std=0.02)
        trunc_normal_(self.mask_token, std=0.02)

        trunc_normal_(self.music_feat_proj.weight, std=0.02)
        nn.init.zeros_(self.music_feat_proj.bias)

        trunc_normal_(self.eeg_linear.weight, std=0.02)
        nn.init.zeros_(self.eeg_linear.bias)

        trunc_normal_(self.proj_out.weight, std=0.02)
        nn.init.zeros_(self.proj_out.bias)

    @torch.jit.ignore
    def no_weight_decay(self):
        """Exclude embeddings and LayerScale parameters from weight decay.
        
        Constructs and returns a set of parameter names that should be excluded 
        from weight decay, which are not automatically excluded by timm.optim.add_weight_decay().

        * Learned embeddings:
          - mask_token
          - eeg_ch_emb  (128 channels)
          - sec_emb     (3 seconds)
          - time20_emb  (150 frames)

        * LayerScale γ parameters (two per Block in encoder/decoder):
          - encoder.{i}.gamma_1 / gamma_2
          - decoder.{i}.gamma_1 / gamma_2

        NOTE: LayerNorm/GroupNorm 1-D weight and bias parameters are already 
              automatically excluded by timm, so they are not listed here.
        """
        names = {
            "mask_token",
            "eeg_ch_emb",
            "sec_emb",
            "time20_emb",
        }
        for i in range(len(self.encoder)):
            names.update({f"encoder.{i}.gamma_1", f"encoder.{i}.gamma_2"})
        for i in range(len(self.decoder)):
            names.update({f"decoder.{i}.gamma_1", f"decoder.{i}.gamma_2"})
        names.update({"cls_token"})
        return names

    def forward(self, eeg_raw: torch.Tensor, Surp_or_Entropy_raw: torch.Tensor = None, ch_name_list=None):
        """
        eeg_raw: (B,128,3,125)
        ch_name_list: optional list of channel names
        """
        B = eeg_raw.size(0)

        # 1. Patch + Linear
        z = self.patch_eeg(eeg_raw)                # (B,384,128)
        z = self.eeg_linear(z)                     # (B,384,512)

        # 2. + Embeddings (ch, sec)
        z = z.view(B, 128, 3, -1)                  # (B,128,3,512)
        idx = list(range(128))
        z = z + self.eeg_ch_emb[idx].unsqueeze(1)
        z = z + self.sec_emb.unsqueeze(0).unsqueeze(0)
        z = z.view(B, 384, -1)                     # (B,384,512)

        cls_tok = self.cls_token.expand(B, -1, -1) # (B,1,512)
        s_enc = torch.cat([cls_tok, z], dim=1)     # (B,384+1,512)

        # 3. Encoder
        h_enc = self.encoder(s_enc)                # (B,384+1,512)
        h_cls = h_enc[:, 0, :]                     # (B,512)

        # 4. Mask tokens (learnable + 20 ms pos-emb)
        time20_emb = self.time20_emb.unsqueeze(0).expand(B, -1, -1)  # (B,150,512)
        mask_tok = self.mask_token.unsqueeze(0).expand(B, 150, -1)   # (B,150,512)

        if Surp_or_Entropy_raw is None:
            raise ValueError("muq_raw is required but was None")

        # 5. Randomly replace 50% of Surp_or_Entropy embeddings with mask tokens
        num_tokens = Surp_or_Entropy_raw.size(1)   # 150
        num_replace = num_tokens // 2              # 50%

        replace_mask = torch.zeros((B, num_tokens), dtype=torch.bool, device=Surp_or_Entropy_raw.device)
        random_scores = torch.rand(B, num_tokens, device=Surp_or_Entropy_raw.device)
        topk_indices = random_scores.topk(num_replace, dim=1).indices
        replace_mask.scatter_(1, topk_indices, True)

        feat_in = Surp_or_Entropy_raw
        if feat_in.ndim == 2:                      # (B,150)
            feat_in = feat_in.unsqueeze(-1)        # (B,150,1)
        elif feat_in.ndim == 3 and feat_in.shape[-1] == 1:
            pass
        else:
            raise ValueError(f"Unexpected shape {feat_in.shape}")

        proj_feat = self.music_feat_proj(feat_in)  # (B,150,512)
        final_tok = torch.where(replace_mask.unsqueeze(-1), proj_feat, mask_tok) + time20_emb

        h_dec = self.decoder(torch.cat([h_enc, final_tok], dim=1))   # (B,384+1 +150,512)
        h_mask = h_dec[:, -150:, :]                                  # (B,150,512)
        out = self.proj_out(h_mask)                                  # (B,150,128)

        mask_pos = ~replace_mask                                      # (B,150) mask_token positions

        return out, h_cls, mask_pos

    def encode_only(self, eeg_raw: torch.Tensor, ch_name_list=None):
        """
        Returns encoder output token sequence (CLS+384) without using decoder.
        Inputs:
            - eeg_raw: (B,128,3,125)
            - ch_name_list: Unused (for future channel selection extension)
        Outputs:
            - h_enc: (B,384+1,512)
        """
        B = eeg_raw.size(0)

        z = self.patch_eeg(eeg_raw)                # (B,384,128)
        z = self.eeg_linear(z)                     # (B,384,512)

        z = z.view(B, 128, 3, -1)
        idx = list(range(128))
        z = z + self.eeg_ch_emb[idx].unsqueeze(1)
        z = z + self.sec_emb.unsqueeze(0).unsqueeze(0)
        z = z.view(B, 384, -1)

        cls_tok = self.cls_token.expand(B, -1, -1)
        s_enc = torch.cat([cls_tok, z], dim=1)     # (B,385,512)
        h_enc = self.encoder(s_enc)                # (B,384+1,512)

        return h_enc


# ----------------------------------------------------------------------
#  (D)  timm.register_model Registration
# ----------------------------------------------------------------------
@register_model
def ms20_comp1_pretrain_ed_2layer_512(pretrained: bool = False, init_ckpt: str = "", **kwargs):
    """
    timm.create_model(
        'ms20_comp1_pretrain_ed_2layer_512',
        embed_dim=512,
        num_heads=8,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_norm=partial(nn.LayerNorm, eps=1e-6),
        qk_scale=None,
        drop_rate=0.1,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        init_values=0.1,
        depth_enc=2,
        depth_dec=2,
        window_size=None,
        attn_head_dim=None,
        music_use_common_time_embed=True
    )
    """
    model = Comp1EDPretrain(**kwargs)
    if pretrained and init_ckpt:
        ckpt = torch.load(init_ckpt, map_location="cpu")
        model.load_state_dict(ckpt["model"], strict=True)
        print(f"[ms20_comp1_pretrain_ed_2layer_512] loaded from {init_ckpt}")
    return model
