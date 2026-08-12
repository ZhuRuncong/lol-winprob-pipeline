"""Baseline 2: L2 logistic regression on the hand-feature frame plus
(diff-feature x time-bucket) interactions, train-standardized.

C is tuned by grouped 5-fold CV (splits.grouped_folds — series never
straddle folds) scored by per-game macro log-loss. Every row carries weight
1/T_g so each game counts equally in the fit.

Usage:
  python -m pipeline.winprob.baselines.logistic \
      --bundles data/winprob_bundles_local --out data/winprob_results_local
"""
from __future__ import annotations

import argparse

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from ..data.splits import grouped_folds
from ..eval.metrics import ABS_LABELS, N_ABS, abs_bucket_index, game_logloss
from .features import (DIFF_FEATURES, build_features, bundle_path, load_bundle,
                       manifest_rows, write_preds)

C_GRID = [0.01, 0.1, 1.0, 10.0]


def design_matrix(bundle: dict) -> tuple[np.ndarray, list[str]]:
    """Base frame + (diff x absolute-time-bucket one-hot) interaction columns."""
    X, names = build_features(bundle)
    T = X.shape[0]
    onehot = np.eye(N_ABS, dtype=np.float32)[abs_bucket_index(np.arange(T))]
    diff_idx = [names.index(n) for n in DIFF_FEATURES]
    inter = (X[:, diff_idx, None] * onehot[:, None, :]).reshape(T, -1)
    inames = [f"{n}_x_{lab}" for n in DIFF_FEATURES for lab in ABS_LABELS]
    return np.concatenate([X, inter], axis=1), names + inames


def load_split_features(bundles_dir, rows) -> dict[str, np.ndarray]:
    return {row["game"]: design_matrix(load_bundle(bundle_path(bundles_dir, row)))[0]
            for row in rows}


def stack(rows, feats) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    X = np.concatenate([feats[r["game"]] for r in rows])
    y = np.concatenate([np.full(len(feats[r["game"]]), r["y"], dtype=np.float32)
                        for r in rows])
    w = np.concatenate([np.full(len(feats[r["game"]]), 1.0 / r["T"]) for r in rows])
    return X, y, w


def fit(rows, feats, C) -> tuple[StandardScaler, LogisticRegression]:
    X, y, w = stack(rows, feats)
    scaler = StandardScaler().fit(X)
    # L2 penalty is sklearn's default (passing penalty= explicitly is deprecated)
    clf = LogisticRegression(C=C, solver="lbfgs", max_iter=2000)
    clf.fit(scaler.transform(X), y, sample_weight=w)
    return scaler, clf


def macro_logloss(rows, feats, scaler, clf) -> float:
    lls = [game_logloss(clf.predict_proba(scaler.transform(feats[r["game"]]))[:, 1],
                        r["y"]) for r in rows]
    return float(np.mean(lls))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bundles", default="data/winprob_bundles_local")
    ap.add_argument("--out", default="data/winprob_results_local")
    args = ap.parse_args()

    rows = manifest_rows(args.bundles)
    train_rows = rows["train"]
    feats = load_split_features(args.bundles, train_rows)
    folds = grouped_folds(train_rows, k=5)

    cv = {}
    for C in C_GRID:
        scores = []
        for tr, va in folds:
            scaler, clf = fit(tr, feats, C)
            scores.append(macro_logloss(va, feats, scaler, clf))
        cv[C] = float(np.mean(scores))
        print(f"  C={C:<5} cv per-game logloss={cv[C]:.4f}")
    best_C = min(cv, key=cv.get)

    scaler, clf = fit(train_rows, feats, best_C)
    preds_by_split = {}
    for split in ("val", "test"):
        sfeats = load_split_features(args.bundles, rows[split])
        preds_by_split[split] = {
            g: clf.predict_proba(scaler.transform(X))[:, 1].astype(np.float32)
            for g, X in sfeats.items()}

    config = {"C": best_C, "C_grid": C_GRID, "cv_logloss": cv,
              "n_features": clf.coef_.shape[1],
              "n_train_games": len(train_rows)}
    out = write_preds(args.out, "logistic", args.bundles, config, preds_by_split)
    print(f"logistic: best C={best_C} (cv={cv[best_C]:.4f}) -> {out}")


if __name__ == "__main__":
    main()
