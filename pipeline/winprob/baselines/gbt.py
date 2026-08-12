"""Baseline 3, the bar to beat: LightGBM on the full hand-feature frame.

Trained on every-5th-second rows (offset 0) with weight 1/ceil(T_g/5) so
each game carries total weight 1. Hyperparameters and early stopping use
GROUPED validation only (splits.grouped_folds): rows within a game are
autocorrelated, so a random row split would leak. Config picked by mean CV
per-game log-loss; refit on all train rows with the CV-mean best_iteration;
predicts every second of val/test.

Usage:
  python -m pipeline.winprob.baselines.gbt \
      --bundles data/winprob_bundles_local --out data/winprob_results_local
"""
from __future__ import annotations

import argparse
import math

import lightgbm as lgb
import numpy as np

from ..data.splits import grouped_folds
from ..eval.metrics import game_logloss
from .features import (build_features, bundle_path, load_bundle,
                       manifest_rows, write_preds)

STRIDE = 5
MAX_ROUNDS = 3000
EARLY_STOP = 50
BASE_PARAMS = {
    "objective": "binary", "metric": "binary_logloss",
    "learning_rate": 0.05, "feature_fraction": 0.9,
    "bagging_fraction": 0.8, "bagging_freq": 1,
    "verbosity": -1, "seed": 0,
}
GRID = [{"num_leaves": nl, "min_data_in_leaf": mdl}
        for nl in (31, 63) for mdl in (100, 300)]


def load_split_features(bundles_dir, rows) -> dict[str, np.ndarray]:
    feats, names = {}, None
    for row in rows:
        X, names = build_features(load_bundle(bundle_path(bundles_dir, row)))
        feats[row["game"]] = X
    return feats, names


def strided(rows, feats) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Every-5th-second training rows; per-game total weight 1."""
    Xs, ys, ws = [], [], []
    for r in rows:
        X = feats[r["game"]][::STRIDE]
        Xs.append(X)
        ys.append(np.full(len(X), r["y"], dtype=np.float32))
        ws.append(np.full(len(X), 1.0 / math.ceil(r["T"] / STRIDE)))
    return np.concatenate(Xs), np.concatenate(ys), np.concatenate(ws)


def cv_config(params, folds, feats, names) -> tuple[float, float]:
    """Mean per-game macro log-loss and mean best_iteration over folds."""
    scores, iters = [], []
    for tr, va in folds:
        Xt, yt, wt = strided(tr, feats)
        Xv, yv, wv = strided(va, feats)
        dtrain = lgb.Dataset(Xt, label=yt, weight=wt, feature_name=names)
        dval = lgb.Dataset(Xv, label=yv, weight=wv, feature_name=names,
                           reference=dtrain)
        booster = lgb.train(params, dtrain, num_boost_round=MAX_ROUNDS,
                            valid_sets=[dval],
                            callbacks=[lgb.early_stopping(EARLY_STOP, verbose=False)])
        iters.append(booster.best_iteration)
        lls = [game_logloss(booster.predict(feats[r["game"]][::STRIDE],
                                            num_iteration=booster.best_iteration),
                            r["y"]) for r in va]
        scores.append(float(np.mean(lls)))
    return float(np.mean(scores)), float(np.mean(iters))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bundles", default="data/winprob_bundles_local")
    ap.add_argument("--out", default="data/winprob_results_local")
    args = ap.parse_args()

    rows = manifest_rows(args.bundles)
    train_rows = rows["train"]
    feats, names = load_split_features(args.bundles, train_rows)
    folds = grouped_folds(train_rows, k=5)

    results = []
    for grid in GRID:
        params = {**BASE_PARAMS, **grid}
        score, best_iter = cv_config(params, folds, feats, names)
        results.append((score, best_iter, grid))
        print(f"  {grid} cv per-game logloss={score:.4f} best_iter~{best_iter:.0f}")
    score, best_iter, grid = min(results, key=lambda r: r[0])
    n_rounds = max(int(round(best_iter)), 1)

    Xt, yt, wt = strided(train_rows, feats)
    dtrain = lgb.Dataset(Xt, label=yt, weight=wt, feature_name=names)
    booster = lgb.train({**BASE_PARAMS, **grid}, dtrain, num_boost_round=n_rounds)

    preds_by_split = {}
    for split in ("val", "test"):
        sfeats, _ = load_split_features(args.bundles, rows[split])
        preds_by_split[split] = {                       # full T, never strided
            g: booster.predict(X).astype(np.float32) for g, X in sfeats.items()}

    config = {**BASE_PARAMS, **grid, "stride": STRIDE, "num_boost_round": n_rounds,
              "cv_logloss": score, "grid": GRID, "early_stopping": EARLY_STOP,
              "n_train_games": len(train_rows)}
    out = write_preds(args.out, "gbt", args.bundles, config, preds_by_split)
    print(f"gbt: best {grid} rounds={n_rounds} (cv={score:.4f}) -> {out}")


if __name__ == "__main__":
    main()
