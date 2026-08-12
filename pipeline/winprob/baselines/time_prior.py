"""Baseline 1: p(blue wins | absolute-time bucket), no game state at all.

Fit on train games with Laplace smoothing; each game contributes total
weight 1 (1/T_g per second) so long games do not dominate a bucket. This is
the floor every stateful model must clear.

Usage:
  python -m pipeline.winprob.baselines.time_prior \
      --bundles data/winprob_bundles_local --out data/winprob_results_local
"""
from __future__ import annotations

import argparse

import numpy as np

from ..eval.metrics import ABS_LABELS, N_ABS, abs_bucket_index
from .features import manifest_rows, write_preds

ALPHA = 1.0


def fit_time_prior(train_rows: list[dict], alpha: float = ALPHA) -> np.ndarray:
    weight = np.zeros(N_ABS)
    wins = np.zeros(N_ABS)
    for row in train_rows:
        # per-game weight 1: each of the T_g seconds carries 1/T_g
        counts = np.bincount(abs_bucket_index(np.arange(row["T"])),
                             minlength=N_ABS) / row["T"]
        weight += counts
        wins += row["y"] * counts
    return (wins + alpha) / (weight + 2.0 * alpha)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bundles", default="data/winprob_bundles_local")
    ap.add_argument("--out", default="data/winprob_results_local")
    args = ap.parse_args()

    rows = manifest_rows(args.bundles)
    prior = fit_time_prior(rows["train"])
    for lab, p in zip(ABS_LABELS, prior):
        print(f"  {lab:>6} min: p(blue)={p:.4f}")

    preds_by_split = {}
    for split in ("val", "test"):
        preds_by_split[split] = {
            row["game"]: prior[abs_bucket_index(np.arange(row["T"]))].astype(np.float32)
            for row in rows[split]}

    config = {"alpha": ALPHA, "buckets": ABS_LABELS,
              "bucket_p": [float(p) for p in prior],
              "n_train_games": len(rows["train"])}
    out = write_preds(args.out, "time_prior", args.bundles, config, preds_by_split)
    print(f"time_prior: fit on {len(rows['train'])} train games -> {out}")


if __name__ == "__main__":
    main()
