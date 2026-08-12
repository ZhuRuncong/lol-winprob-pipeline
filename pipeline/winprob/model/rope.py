"""Rotary position embeddings, plain torch.

ROCm portability: no fused RoPE kernels, no CUDA-only ops — just the
half-split rotation (GPT-NeoX convention) built from arange/cos/sin, so the
same code runs on CPU tests and on a ROCm build unchanged. Positions are the
absolute second index (all games start at t=0 and are right-padded).
"""
from __future__ import annotations

import torch


def rope_cache(T: int, head_dim: int, theta: float, device: torch.device,
               dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    """cos/sin tables for absolute positions 0..T-1, each [T, head_dim//2]."""
    half = head_dim // 2
    inv_freq = 1.0 / (theta ** (
        torch.arange(0, half, device=device, dtype=torch.float32) / half))
    ang = torch.arange(T, device=device, dtype=torch.float32)[:, None] * inv_freq[None, :]
    return ang.cos().to(dtype), ang.sin().to(dtype)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate q or k. x [B, H, T, head_dim]; cos/sin [T, head_dim//2]."""
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)
