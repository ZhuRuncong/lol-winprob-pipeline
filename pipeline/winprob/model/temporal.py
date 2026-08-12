"""Causal temporal trunk: pre-norm GPT blocks over the 1 Hz state sequence.

ROCm portability: plain-torch RoPE (rope.py) plus
F.scaled_dot_product_attention only — no flash-attn, no xformers, no fused or
CUDA-only kernels. Runs unchanged on CPU.

Padding: sequences are right-padded and every game starts at t=0, so under
is_causal=True no real token can attend a pad token (pads sit strictly in the
future of every real query). Hence there is NO key-padding mask here; `valid`
is accepted for API symmetry and pad positions are excluded by the losses.

Ablation flags (config fields, never code forks):
  attn_window > 0  -> additionally mask keys older than attn_window seconds
                      (bool band mask, True = may attend, built once per
                      forward via tril/triu).
  context_one      -> no attention sublayers at all; per-second MLP blocks.

Stochastic depth ramps linearly 0 -> cfg.stoch_depth across layers; the
residual branch is dropped per-sample during training.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from ..config import ModelConfig
from .rope import apply_rope, rope_cache


class DropPath(nn.Module):
    """Per-sample residual-branch drop (stochastic depth)."""

    def __init__(self, p: float):
        super().__init__()
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.p <= 0.0:
            return x
        keep = 1.0 - self.p
        mask = x.new_empty(x.shape[0], *([1] * (x.ndim - 1))).bernoulli_(keep)
        return x * mask / keep


class TrunkBlock(nn.Module):
    def __init__(self, cfg: ModelConfig, drop_path: float):
        super().__init__()
        d = cfg.d_model
        self.use_attn = not cfg.context_one
        self.dropout = cfg.dropout
        if self.use_attn:
            self.heads = cfg.heads
            self.ln1 = nn.LayerNorm(d)
            self.qkv = nn.Linear(d, 3 * d)
            self.proj = nn.Linear(d, d)
        self.ln2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(
            nn.Linear(d, cfg.mlp_ratio * d), nn.GELU(), nn.Dropout(cfg.dropout),
            nn.Linear(cfg.mlp_ratio * d, d))
        self.drop = nn.Dropout(cfg.dropout)
        self.drop_path = DropPath(drop_path)

    def forward(self, x: torch.Tensor, cos: torch.Tensor | None,
                sin: torch.Tensor | None,
                attn_mask: torch.Tensor | None) -> torch.Tensor:
        if self.use_attn:
            B, T, d = x.shape
            hd = d // self.heads
            q, k, v = self.qkv(self.ln1(x)).chunk(3, dim=-1)
            # [B,T,d] -> [B,heads,T,hd]
            q = q.view(B, T, self.heads, hd).transpose(1, 2)
            k = k.view(B, T, self.heads, hd).transpose(1, 2)
            v = v.view(B, T, self.heads, hd).transpose(1, 2)
            q = apply_rope(q, cos, sin)
            k = apply_rope(k, cos, sin)
            p = self.dropout if self.training else 0.0
            if attn_mask is None:
                a = F.scaled_dot_product_attention(q, k, v, dropout_p=p, is_causal=True)
            else:
                a = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=p)
            a = a.transpose(1, 2).reshape(B, T, d)
            x = x + self.drop_path(self.drop(self.proj(a)))
        x = x + self.drop_path(self.drop(self.mlp(self.ln2(x))))
        return x


class CausalTrunk(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        assert cfg.d_model % cfg.heads == 0, "d_model must divide heads"
        self.head_dim = cfg.d_model // cfg.heads
        assert self.head_dim % 2 == 0, "RoPE needs an even head dim"
        self.cfg = cfg
        L = cfg.layers
        ramp = [cfg.stoch_depth * i / max(L - 1, 1) for i in range(L)]
        self.blocks = nn.ModuleList(TrunkBlock(cfg, p) for p in ramp)
        self.ln_f = nn.LayerNorm(cfg.d_model)

    def forward(self, s: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        """s [B,T,d_model], valid [B,T] (unused; see module docstring) -> h [B,T,d_model]."""
        B, T, _ = s.shape
        cos = sin = attn_mask = None
        if not self.cfg.context_one:
            cos, sin = rope_cache(T, self.head_dim, self.cfg.rope_theta,
                                  s.device, s.dtype)
            if self.cfg.attn_window > 0:
                ones = torch.ones(T, T, dtype=torch.bool, device=s.device)
                # True = may attend: causal AND key age < attn_window. The band
                # always contains the diagonal, so no query row is fully masked.
                attn_mask = torch.tril(ones) & torch.triu(
                    ones, diagonal=-(self.cfg.attn_window - 1))
        x = s
        for blk in self.blocks:
            x = blk(x, cos, sin, attn_mask)
        return self.ln_f(x)
