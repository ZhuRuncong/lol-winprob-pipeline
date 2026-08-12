"""Per-second state encoder: 13 tokens -> set transformer -> one vector.

Tokens per second: 10 champion tokens (shared weights, statics concatenated so
the network always knows WHO it is reading), 2 team tokens (ward rasters
included), 1 global token (clock Fourier + patch + swap flag). Two
bidirectional set-attention layers mix them — spatial attention within one
instant, so no causality is needed; the fold to [B*T, 13, d_tok] keeps it one
SDPA call.

Readout: role-ordered concatenation of all 13 token outputs (slot identity is
preserved through the primary path) plus a residual flat skip. Both paths
consume the SAME batch tensors, so fog-of-war masking applied to the batch
upstream reaches every input path — the masking-leak unit test depends on
this property.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from ..config import (CHAMPION_VOCAB, D_CHAMP_CONT, D_GLOBAL_CONT, D_TEAM_CONT,
                      ITEM_HASH_BUCKETS, KEYSTONE_HASH_BUCKETS, ModelConfig,
                      N_ITEM_SLOTS, SPELL_VOCAB, STYLE_VOCAB)

N_TOKENS = 13          # 10 champions + 2 teams + 1 global
D_STATIC_EMB = 32 + 8 + 8 + 8 + 2 + 2 + 2   # champ+role+spells+keystone+styles+side
D_SKIP = D_CHAMP_CONT * 10 + D_TEAM_CONT * 2 + D_GLOBAL_CONT   # 684


class SetBlock(nn.Module):
    """Pre-norm bidirectional transformer layer over the 13 tokens."""

    def __init__(self, d: int, heads: int, dropout: float):
        super().__init__()
        self.heads = heads
        self.dropout = dropout
        self.ln1 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.ln2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(),
                                 nn.Dropout(dropout), nn.Linear(2 * d, d))
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:      # [N, 13, d]
        N, S, d = x.shape
        hd = d // self.heads
        q, k, v = self.qkv(self.ln1(x)).chunk(3, dim=-1)
        q = q.view(N, S, self.heads, hd).transpose(1, 2)
        k = k.view(N, S, self.heads, hd).transpose(1, 2)
        v = v.view(N, S, self.heads, hd).transpose(1, 2)
        p = self.dropout if self.training else 0.0
        a = F.scaled_dot_product_attention(q, k, v, dropout_p=p)
        x = x + self.drop(self.proj(a.transpose(1, 2).reshape(N, S, d)))
        return x + self.drop(self.mlp(self.ln2(x)))


class StateEncoder(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        dt = cfg.d_tok
        self.champ_emb = nn.Embedding(CHAMPION_VOCAB, 32)
        self.role_emb = nn.Embedding(5, 8)
        self.spell_emb = nn.Embedding(SPELL_VOCAB, 8)
        self.keystone_emb = nn.Embedding(KEYSTONE_HASH_BUCKETS, 8)
        self.style_emb = nn.Embedding(STYLE_VOCAB, 2)
        self.substyle_emb = nn.Embedding(STYLE_VOCAB, 2)
        self.side_emb = nn.Embedding(2, 2)
        self.patch_emb = nn.Embedding(64, 8)
        self.item_emb = nn.Embedding(ITEM_HASH_BUCKETS, cfg.d_item_emb, padding_idx=0)

        self.champ_in = nn.Linear(D_CHAMP_CONT + D_STATIC_EMB + cfg.d_item_emb + 1, dt)
        self.team_in = nn.Linear(D_TEAM_CONT + 2, dt)
        self.glob_in = nn.Linear(D_GLOBAL_CONT + 8 + 1, dt)

        self.set_blocks = nn.ModuleList(
            SetBlock(dt, cfg.set_heads, cfg.dropout) for _ in range(cfg.set_layers))
        self.readout = nn.Linear(N_TOKENS * dt, cfg.d_model)
        self.skip = nn.Linear(D_SKIP, cfg.d_model)
        self.ln_out = nn.LayerNorm(cfg.d_model)

    def forward(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
        champ = batch["champ"].float()               # [B,T,10,58]
        team = batch["team"].float()                 # [B,T,2,46]
        glob = batch["glob"].float()                 # [B,T,12]
        B, T, _, _ = champ.shape

        st = batch["statics"]                        # [B,10,7] int64
        static_vec = torch.cat([
            self.champ_emb(st[..., 0].clamp(0, CHAMPION_VOCAB - 1)),
            self.role_emb(st[..., 1].clamp(0, 4)),
            self.spell_emb(st[..., 2].clamp(0, SPELL_VOCAB - 1))
            + self.spell_emb(st[..., 3].clamp(0, SPELL_VOCAB - 1)),
            self.keystone_emb(st[..., 4].clamp(0, KEYSTONE_HASH_BUCKETS - 1)),
            self.style_emb(st[..., 5].clamp(0, STYLE_VOCAB - 1)),
            self.substyle_emb(st[..., 6].clamp(0, STYLE_VOCAB - 1)),
            self.side_emb(batch["side"]),
        ], dim=-1)                                   # [B,10,62]
        static_vec = static_vec[:, None].expand(B, T, 10, D_STATIC_EMB)

        # item bag: embed, FiLM-scale by cooldown state, mean over slots
        item_e = self.item_emb(batch["items"])       # [B,T,10,7,16]
        item_e = item_e * (1.0 + torch.tanh(batch["item_cd"].float()))[..., None]
        item_vec = item_e.mean(dim=3)                # [B,T,10,16]

        mask_flag = batch.get("champ_mask")
        mask_flag = (mask_flag.float() if mask_flag is not None
                     else champ.new_zeros(B, T, 10))[..., None]

        champ_tok = self.champ_in(torch.cat([champ, static_vec, item_vec, mask_flag], -1))

        team_side = torch.stack([batch["side"][:, 0], batch["side"][:, 5]], 1)  # [B,2]
        team_tok = self.team_in(torch.cat(
            [team, self.side_emb(team_side)[:, None].expand(B, T, 2, 2)], -1))

        glob_tok = self.glob_in(torch.cat([
            glob,
            self.patch_emb(batch["patch_idx"].clamp(0, 63))[:, None].expand(B, T, 8),
            batch["swap"][:, None, None].expand(B, T, 1),
        ], -1))                                      # [B,T,dt]

        toks = torch.cat([champ_tok, team_tok, glob_tok[:, :, None]], dim=2)  # [B,T,13,dt]
        x = toks.view(B * T, N_TOKENS, self.cfg.d_tok)
        for blk in self.set_blocks:
            x = blk(x)
        x = x.view(B, T, N_TOKENS, self.cfg.d_tok)

        C = x[:, :, :10]                             # champion token outputs
        flat = torch.cat([x[:, :, 12], x[:, :, 10], x[:, :, 11],
                          C.reshape(B, T, -1)], dim=-1)
        skip_in = torch.cat([champ.reshape(B, T, -1), team.reshape(B, T, -1), glob], -1)
        s = self.ln_out(self.readout(flat) + self.skip(skip_in))
        return s, C
