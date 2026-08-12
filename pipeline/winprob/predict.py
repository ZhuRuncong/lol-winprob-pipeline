"""Full-sequence causal inference -> standard prediction files.

Prediction contract (shared by baselines, members, ensembles):
  results/<name>/<split>_preds.npz        keys = game names, values = float32 [T] p(blue)
  results/<name>/<split>_preds_swap.npz   same, computed on the team-swapped view
  results/<name>/meta.json

Usage:
  python -m pipeline.winprob.predict --ckpt runs/ft1/best.pt \
      --bundles data/winprob_bundles --out results/member1 --splits val test
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .data.dataset import GameBundleDataset, TokenBudgetBatchSampler, collate
from .train_common import autocast_ctx, load_norm_version, pick_device


class SwappedView(Dataset):
    """Eval-time team-relabeled view of a dataset (for symmetrized inference)."""

    def __init__(self, ds: GameBundleDataset):
        self.ds = ds

    def __len__(self):
        return len(self.ds)

    def lengths(self):
        return self.ds.lengths()

    def __getitem__(self, idx):
        d = self.ds[idx]
        GameBundleDataset._swap(d)
        return d


def load_model(ckpt_path: str | Path, device, use_ema: bool = True):
    from .model import WinProbModel
    from .config import Config, ModelConfig, MaskConfig, LossWeights, TrainConfig, DataConfig
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    c = ck["cfg"]  # rebuild the exact config the checkpoint was trained with
    cfg = Config(model=ModelConfig(**c["model"]), mask=MaskConfig(**c["mask"]),
                 loss=LossWeights(**c["loss"]), train=TrainConfig(**c["train"]),
                 data=DataConfig(**c["data"]), name=c.get("name", "ckpt"))
    model = WinProbModel(cfg.model).to(device)
    model.load_state_dict(ck["model"])
    if use_ema and ck.get("ema"):
        sd = model.state_dict()
        for k, v in ck["ema"].items():
            sd[k].copy_(v.to(sd[k].dtype))
    model.eval()
    return model, cfg, ck.get("norm_version", "?")


@torch.no_grad()
def predict_split(model, ds, cfg, device, budget: int = 65536) -> dict[str, np.ndarray]:
    sampler = TokenBudgetBatchSampler(ds.lengths(), budget=budget, shuffle=False)
    loader = DataLoader(ds, batch_sampler=sampler, collate_fn=collate, num_workers=0)
    rows = ds.ds.rows if isinstance(ds, SwappedView) else ds.rows
    preds: dict[str, np.ndarray] = {}
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with autocast_ctx(device, cfg.train.amp_dtype):
            out = model(batch, ssl=False)
        p = torch.sigmoid(out["win_logit"].float()).cpu().numpy()
        for b in range(p.shape[0]):
            row = rows[int(batch["game_idx"][b])]
            preds[row["game"]] = p[b, :row["T"]].astype(np.float32)
    return preds


def write_preds(out_dir: Path, split: str, preds: dict, swap: bool = False):
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "_preds_swap.npz" if swap else "_preds.npz"
    np.savez(out_dir / f"{split}{suffix}", **preds)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--bundles", default="data/winprob_bundles")
    ap.add_argument("--out", required=True)
    ap.add_argument("--splits", nargs="+", default=["val", "test"])
    ap.add_argument("--no-swap-pass", action="store_true")
    ap.add_argument("--budget", type=int, default=65536)
    args = ap.parse_args()

    device = pick_device()
    model, cfg, norm_version = load_model(args.ckpt, device)
    bundles_norm = load_norm_version(args.bundles)
    if norm_version not in ("?", bundles_norm):
        raise SystemExit(f"norm mismatch: checkpoint {norm_version} vs bundles {bundles_norm}")

    out = Path(args.out)
    for split in args.splits:
        ds = GameBundleDataset(args.bundles, split)
        write_preds(out, split, predict_split(model, ds, cfg, device, args.budget))
        if not args.no_swap_pass:
            write_preds(out, split, predict_split(model, SwappedView(ds), cfg, device,
                                                  args.budget), swap=True)
        print(f"wrote {split} predictions ({len(ds)} games)")

    with open(out / "meta.json", "w", encoding="utf-8") as fh:
        json.dump({"model": str(args.ckpt), "bundles": str(args.bundles),
                   "norm_version": bundles_norm,
                   "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")},
                  fh, indent=1)


if __name__ == "__main__":
    main()
