"""The single source of truth for data grouping.

Every consumer of train/val/test membership, CV folds, or bootstrap
resampling units imports from THIS module. Nothing else may decide grouping:
timesteps are never independent samples, games of one series never straddle
splits.
"""
from __future__ import annotations

import json
from pathlib import Path

SPLITS = ("train", "val", "test")


def load_manifest(corpus_root: str | Path) -> dict:
    with open(Path(corpus_root) / "manifest.json", encoding="utf-8") as fh:
        return json.load(fh)


def split_games(manifest: dict) -> dict[str, list[dict]]:
    """Manifest -> {split: [game rows]} using the published series-grouped split."""
    out: dict[str, list[dict]] = {s: [] for s in SPLITS}
    for g in manifest["games"]:
        out[g["split"]].append(g)
    return out


def series_of(manifest: dict) -> dict[str, str]:
    return {g["game"]: g["series_id"] for g in manifest["games"]}


def grouped_folds(rows: list[dict], k: int = 5, seed: int = 0) -> list[tuple[list[dict], list[dict]]]:
    """K-fold CV over game rows, grouped by series_id.

    For baseline hyperparameter tuning and early stopping: all games of a
    series land in the same fold.
    """
    import random
    series = sorted({g["series_id"] for g in rows})
    rng = random.Random(seed)
    rng.shuffle(series)
    fold_of = {s: i % k for i, s in enumerate(series)}
    folds = []
    for f in range(k):
        tr = [g for g in rows if fold_of[g["series_id"]] != f]
        va = [g for g in rows if fold_of[g["series_id"]] == f]
        folds.append((tr, va))
    return folds


def subsample_series(rows: list[dict], fraction: float, seed: int = 0) -> list[dict]:
    """Keep a seeded fraction of SERIES (whole series in or out) — the
    labeled-corpus-fraction ablation arm ('SSL advantage must grow as labeled
    games shrink')."""
    import random
    series = sorted({g["series_id"] for g in rows})
    rng = random.Random(seed)
    rng.shuffle(series)
    keep = set(series[: max(1, round(len(series) * fraction))])
    return [g for g in rows if g["series_id"] in keep]


def patch_key(patch: str) -> tuple:
    """Sortable version tuple from a patch string like '16.10.776.5552'."""
    parts = []
    for p in str(patch).split("."):
        try:
            parts.append(int(p))
        except ValueError:
            parts.append(0)
    return tuple(parts[:2])


def temporal_split(manifest: dict, holdout_patches: int = 2) -> dict[str, list[dict]]:
    """Committed secondary split: hold out the final N patches entirely.

    Returns {"train": rows, "test": rows}. The grouped-vs-temporal metric gap
    is itself a reported number (patch-drift exposure).
    """
    rows = manifest["games"]
    patches = sorted({patch_key(g["patch"]) for g in rows})
    if len(patches) <= holdout_patches:
        raise ValueError(f"only {len(patches)} distinct patches; cannot hold out {holdout_patches}")
    held = set(patches[-holdout_patches:])
    return {
        "train": [g for g in rows if patch_key(g["patch"]) not in held],
        "test": [g for g in rows if patch_key(g["patch"]) in held],
    }


def temporal_split_with_val(manifest: dict, holdout_patches: int = 2,
                            val_frac: float = 0.1, seed: int = 0) -> dict[str, list[dict]]:
    """Temporal split plus a series-grouped val carve-out from temporal-train
    (early stopping / calibration need a val set that predates the holdout)."""
    import random
    tr_te = temporal_split(manifest, holdout_patches)
    series = sorted({g["series_id"] for g in tr_te["train"]})
    rng = random.Random(seed)
    rng.shuffle(series)
    val_set = set(series[: max(1, round(len(series) * val_frac))])
    return {
        "train": [g for g in tr_te["train"] if g["series_id"] not in val_set],
        "val": [g for g in tr_te["train"] if g["series_id"] in val_set],
        "test": tr_te["test"],
    }
