"""Fit normalization statistics on the TRAIN split only -> norm_v2.json.

Supersedes the corpus-wide norm.json from the pack stage (which pools
train/val/test — a leakage bug; that file is left untouched and ignored).
Computes: z-score mean/std for the z-masked champion/team dims, per-horizon
future-delta stats for SSL target normalization, and the patch vocabulary.
The output carries a content hash; every downstream artifact asserts it.

Usage:
  python -m pipeline.winprob.data.stats [--local-sample] [--out PATH]
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from ..config import (DELTA_CHAMP_FEATURES, DELTA_HORIZONS, DELTA_TEAM_FEATURES,
                      HF_REVISION, CI, TI)
from . import transforms as tr
from .hub import corpus_root, iter_shard_games
from .splits import load_manifest, split_games, temporal_split_with_val

EPS = 1e-6


class Running:
    """Streaming mean/std with a per-column sample count."""

    def __init__(self, dim):
        self.s = np.zeros(dim, dtype=np.float64)
        self.sq = np.zeros(dim, dtype=np.float64)
        self.n = np.zeros(dim, dtype=np.float64)

    def add(self, a, cols=slice(None)):  # a: [N, n_cols]
        a = a.astype(np.float64)
        self.s[cols] += a.sum(0)
        self.sq[cols] += (a ** 2).sum(0)
        self.n[cols] += a.shape[0]

    def stats(self):
        n = np.maximum(self.n, 1)
        mean = self.s / n
        var = np.maximum(self.sq / n - mean ** 2, EPS)
        return mean, np.sqrt(var)


def fit(local_sample: bool = False, split_source: str = "manifest") -> dict:
    root = corpus_root(local_sample=local_sample)
    manifest = load_manifest(root)
    if split_source == "temporal":
        train_rows = temporal_split_with_val(manifest)["train"]
    else:
        train_rows = split_games(manifest)["train"]
    wanted = {g["game"] for g in train_rows}

    champ = Running(int(tr.CHAMP_Z_MASK.sum()))
    team = Running(int(tr.TEAM_Z_MASK.sum()))
    n_h = len(DELTA_HORIZONS)
    d_champ = Running(len(DELTA_CHAMP_FEATURES) * n_h)
    d_team = Running(len(DELTA_TEAM_FEATURES) * n_h)
    patches: set[str] = set()
    n_games = 0

    for game, meta, npz in iter_shard_games(root, wanted):
        n_games += 1
        X_champ, X_kda, X_team = npz["X_champ"], npz["X_champ_kda"], npz["X_team"]
        T = X_champ.shape[0]

        cb = tr.champ_cont_prenorm(X_champ, X_kda)
        champ.add(cb[:, :, tr.CHAMP_Z_MASK].reshape(-1, champ.s.shape[0]))
        tb = tr.team_cont_prenorm(X_team, npz["wards"], T)
        team.add(tb[:, :, tr.TEAM_Z_MASK].reshape(-1, team.s.shape[0]))

        # per-horizon future deltas, pooled over champs / teams
        # column layout: index = feature_index * n_horizons + horizon_index
        gold_diff = X_team[:, 0, TI["total_gold"]] - X_team[:, 1, TI["total_gold"]]
        for hi, h in enumerate(DELTA_HORIZONS):
            if T <= h:
                continue
            cd = np.stack([X_champ[h:, :, CI[f]] - X_champ[:-h, :, CI[f]]
                           for f in DELTA_CHAMP_FEATURES], axis=-1)  # [T-h,10,3]
            d_champ.add(cd.reshape(-1, len(DELTA_CHAMP_FEATURES)), cols=slice(hi, None, n_h))

            td = np.stack([
                gold_diff[h:] - gold_diff[:-h],
                (X_team[h:, :, TI["towers_destroyed"]] -
                 X_team[:-h, :, TI["towers_destroyed"]]).sum(-1),
            ], axis=-1)  # [T-h, 2]
            d_team.add(td, cols=slice(hi, None, n_h))

        patches.add(tr.parse_patch(meta.get("patch")))
        if n_games % 250 == 0:
            print(f"  {n_games} train games streamed", flush=True)

    champ_mean, champ_std = champ.stats()
    team_mean, team_std = team.stats()
    dc_mean, dc_std = d_champ.stats()
    dt_mean, dt_std = d_team.stats()

    patch_vocab = {p: i + 1 for i, p in enumerate(sorted(patches))}  # 0 = OOV

    stats = {
        "split_source": split_source,
        "hf_revision": HF_REVISION if not local_sample else "local-sample",
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "n_train_games": n_games,
        "champ_mean": champ_mean.tolist(), "champ_std": champ_std.tolist(),
        "team_mean": team_mean.tolist(), "team_std": team_std.tolist(),
        "delta_horizons": DELTA_HORIZONS,
        "delta_champ_features": DELTA_CHAMP_FEATURES,
        "delta_team_features": DELTA_TEAM_FEATURES,
        "delta_champ_mean": dc_mean.tolist(), "delta_champ_std": dc_std.tolist(),
        "delta_team_mean": dt_mean.tolist(), "delta_team_std": dt_std.tolist(),
        "patch_vocab": patch_vocab,
    }
    payload = json.dumps(stats, sort_keys=True).encode()
    stats["norm_version"] = "v2-" + hashlib.sha256(payload).hexdigest()[:8]
    return stats


def load_stats(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as fh:
        s = json.load(fh)
    if not str(s.get("norm_version", "")).startswith("v2-"):
        raise SystemExit(f"{path}: not a norm_v2 stats file (train-split-only); refusing")
    return s


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--local-sample", action="store_true")
    ap.add_argument("--split", choices=["manifest", "temporal"], default="manifest",
                    help="'temporal' fits stats on the late-patch-holdout train set "
                         "(pack must use the same --split; use a separate --out dir)")
    ap.add_argument("--out", default="data/winprob_bundles/norm_v2.json")
    args = ap.parse_args()

    stats = fit(local_sample=args.local_sample, split_source=args.split)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(stats, fh)
    print(f"wrote {out}  norm_version={stats['norm_version']}  "
          f"n_train_games={stats['n_train_games']}  patches={len(stats['patch_vocab'])}")


if __name__ == "__main__":
    main()
