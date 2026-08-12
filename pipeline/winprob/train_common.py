"""Shared training machinery: device/AMP, EMA, param groups, schedules,
checkpoints, run ledger. Used by both training phases."""
from __future__ import annotations

import json
import math
import subprocess
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import torch


def pick_device() -> torch.device:
    # ROCm builds expose the GPU through the cuda API
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def autocast_ctx(device: torch.device, amp_dtype: str):
    if device.type != "cuda" or amp_dtype == "fp32":
        import contextlib
        return contextlib.nullcontext()
    dtype = torch.bfloat16 if amp_dtype == "bf16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


def assert_sdpa_backend(heads: int = 12, head_dim: int = 32, seq: int = 2048,
                        dropout_p: float = 0.1):
    """Probe which SDPA backends actually RUN a production-shaped causal call.

    The global 'enabled' flags don't prove a backend accepts our exact
    (dtype, head_dim, dropout, causal) combination — ROCm builds silently
    fall back to the math kernel. Executes a real call under each backend and
    warns loudly if only MATH works (training would be correct but slow).
    """
    if not torch.cuda.is_available():
        return "cpu"
    from torch.nn.attention import SDPBackend, sdpa_kernel
    q = torch.randn(1, heads, seq, head_dim, device="cuda", dtype=torch.bfloat16)
    works = {}
    for name, backend in (("flash", SDPBackend.FLASH_ATTENTION),
                          ("mem_efficient", SDPBackend.EFFICIENT_ATTENTION),
                          ("math", SDPBackend.MATH)):
        try:
            with sdpa_kernel([backend]):
                torch.nn.functional.scaled_dot_product_attention(
                    q, q, q, dropout_p=dropout_p, is_causal=True)
            works[name] = True
        except Exception:
            works[name] = False
    print(f"SDPA backend probe (bf16, h{heads}x{head_dim}, T={seq}, "
          f"dropout={dropout_p}, causal): {works}")
    if not any(works.values()):
        raise RuntimeError("no SDPA backend can run the production attention shape")
    if not (works["flash"] or works["mem_efficient"]):
        print("WARNING: only the MATH backend works — attention will be slow; "
              "consider dropout_p=0 in attention or a different ROCm/torch tag")
    return works


def param_groups(model: torch.nn.Module, weight_decay: float):
    """Weight decay on matrices only — never embeddings, norms, or biases."""
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim < 2 or "emb" in name.lower() or "norm" in name.lower():
            no_decay.append(p)
        else:
            decay.append(p)
    return [{"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0}]


def cosine_lr(step: int, total_steps: int, warmup_steps: int, base_lr: float,
              floor_frac: float = 0.1) -> float:
    if step < warmup_steps:
        return base_lr * (step + 1) / max(warmup_steps, 1)
    prog = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    return base_lr * (floor_frac + (1 - floor_frac) * 0.5 * (1 + math.cos(math.pi * min(prog, 1.0))))


class EMA:
    def __init__(self, model: torch.nn.Module, decay: float):
        self.decay = decay
        self.shadow = {k: v.detach().clone().float()
                       for k, v in model.state_dict().items() if v.dtype.is_floating_point}

    @torch.no_grad()
    def update(self, model: torch.nn.Module):
        for k, v in model.state_dict().items():
            if k in self.shadow:
                self.shadow[k].mul_(self.decay).add_(v.detach().float(), alpha=1 - self.decay)

    def copy_to(self, model: torch.nn.Module) -> dict:
        """Swap EMA weights in; returns the originals for swap-back."""
        backup = {}
        sd = model.state_dict()
        for k, shadow in self.shadow.items():
            backup[k] = sd[k].detach().clone()
            sd[k].copy_(shadow.to(sd[k].dtype))
        return backup

    @staticmethod
    def restore(model: torch.nn.Module, backup: dict):
        sd = model.state_dict()
        for k, v in backup.items():
            sd[k].copy_(v)


def git_sha() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, timeout=10,
                              cwd=Path(__file__).parent).stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def save_checkpoint(path: Path, model, ema: EMA | None, opt, epoch: int, cfg,
                    norm_version: str, extra: dict | None = None):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model": model.state_dict(),
        "ema": ema.shadow if ema else None,
        "opt": opt.state_dict() if opt else None,
        "epoch": epoch,
        "cfg": asdict(cfg),
        "norm_version": norm_version,
        "git_sha": git_sha(),
        "saved_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        **(extra or {}),
    }, path)


def load_norm_version(bundles_dir: str | Path) -> str:
    with open(Path(bundles_dir) / "bundles_manifest.json", encoding="utf-8") as fh:
        return json.load(fh)["norm_version"]


class MetricsLog:
    """Append-only JSONL metrics ledger inside a run directory."""

    def __init__(self, run_dir: Path):
        run_dir.mkdir(parents=True, exist_ok=True)
        self.path = run_dir / "metrics.jsonl"

    def log(self, **row):
        row.setdefault("utc", datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(row) + "\n")
        keys = [k for k in row if k not in ("utc",)]
        print("  " + "  ".join(f"{k}={row[k]:.5g}" if isinstance(row[k], float)
                               else f"{k}={row[k]}" for k in keys), flush=True)
