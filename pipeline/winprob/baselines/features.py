"""Per-second hand-feature frames shared by the logistic and GBT baselines.

Built from the RAW target arrays (tgt_champ [T,10,7], tgt_team [T,2,5]) so
the baselines stay interpretable and carry no normalization dependency.
Everything is strictly causal: every column at second t reads only values at
times <= t; trailing windows clamp at the game start and divide by the
actual elapsed time. Also holds the shared bundle/manifest IO and the
predictions-directory writer (the project-wide predictions contract).
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from ..data.transforms import OFF_MICRO

SLOPE_WINDOWS = [30, 60, 120]

# the transformer's causal micro-deltas (bundle champ block cols 46:58,
# z-scored; layout: rates w in {30,60,90} x {cs,gold,xp}, then disp10/disp30,
# then hp_frac delta-5s) — fed to the GBT too so the null is honest
MICRO_NAMES = [f"{f}_rate_{w}s" for w in (30, 60, 90) for f in ("cs", "gold", "xp")] \
    + ["disp_10s", "disp_30s", "hp_frac_d5s"]

# blue-minus-red diff columns; the logistic baseline crosses exactly these
# with time buckets.
DIFF_FEATURES = [
    "diff_gold", "diff_kills", "diff_towers", "diff_dragons", "diff_baron",
    "diff_level_sum", "diff_cs_sum", "diff_xp_sum", "diff_alive",
]

# column layout of the raw target arrays (see transforms.TGT_*_FEATURES)
_C_CS, _C_GOLD, _C_XP, _C_HP, _C_MANA, _C_LEVEL, _C_ALIVE = range(7)
_T_KILLS, _T_TOWERS, _T_DRAGONS, _T_BARON, _T_GOLD = range(5)


def load_bundle(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def manifest_rows(bundles_dir: str | Path) -> dict[str, list[dict]]:
    """bundles_manifest.json -> {split: [game rows]} (rows carry series_id)."""
    with open(Path(bundles_dir) / "bundles_manifest.json", encoding="utf-8") as fh:
        manifest = json.load(fh)
    out: dict[str, list[dict]] = {}
    for g in manifest["games"]:
        out.setdefault(g["split"], []).append(g)
    return out


def bundle_path(bundles_dir: str | Path, row: dict) -> Path:
    return Path(bundles_dir) / row["split"] / f"{row['game']}.npz"


def build_features(bundle: dict) -> tuple[np.ndarray, list[str]]:
    """Bundle dict -> (X [T,D] float32, column names)."""
    champ = bundle["tgt_champ"].astype(np.float32)      # [T,10,7] raw
    team = bundle["tgt_team"].astype(np.float32)        # [T,2,5] raw
    T = champ.shape[0]
    names, cols = [], []

    def add(name, arr):
        names.append(name)
        cols.append(np.asarray(arr, dtype=np.float32))

    blue, red = team[:, 0], team[:, 1]
    diffs = {
        "diff_gold": blue[:, _T_GOLD] - red[:, _T_GOLD],
        "diff_kills": blue[:, _T_KILLS] - red[:, _T_KILLS],
        "diff_towers": blue[:, _T_TOWERS] - red[:, _T_TOWERS],
        "diff_dragons": blue[:, _T_DRAGONS] - red[:, _T_DRAGONS],
        "diff_baron": blue[:, _T_BARON] - red[:, _T_BARON],
        "diff_level_sum": champ[:, :5, _C_LEVEL].sum(1) - champ[:, 5:, _C_LEVEL].sum(1),
        "diff_cs_sum": champ[:, :5, _C_CS].sum(1) - champ[:, 5:, _C_CS].sum(1),
        "diff_xp_sum": champ[:, :5, _C_XP].sum(1) - champ[:, 5:, _C_XP].sum(1),
        "diff_alive": champ[:, :5, _C_ALIVE].sum(1) - champ[:, 5:, _C_ALIVE].sum(1),
    }
    for name in DIFF_FEATURES:
        add(name, diffs[name])

    ts = np.arange(T, dtype=np.float32)
    add("t_seconds", ts)
    add("t_frac_1800", ts / 1800.0)
    add("blue_gold", blue[:, _T_GOLD])
    add("red_gold", red[:, _T_GOLD])
    add("blue_towers", blue[:, _T_TOWERS])
    add("red_towers", red[:, _T_TOWERS])

    ti = np.arange(T)
    for w in SLOPE_WINDOWS:
        back = np.maximum(ti - w, 0)
        elapsed = np.maximum(ti - back, 1).astype(np.float32)
        for name in ("diff_gold", "diff_cs_sum", "diff_xp_sum"):
            d = diffs[name]
            add(f"slope_{name}_{w}s", (d - d[back]) / elapsed)
        for name in ("diff_kills", "diff_towers"):
            d = diffs[name]
            add(f"delta_{name}_{w}s", d - d[back])

    for r, role in enumerate(("top", "jungle", "mid", "bot", "sup")):
        add(f"role_gold_diff_{role}", champ[:, r, _C_GOLD] - champ[:, 5 + r, _C_GOLD])

    def mean_alive_hp(side):
        alive = champ[:, side, _C_ALIVE]
        n = alive.sum(1)
        return np.where(n > 0, (champ[:, side, _C_HP] * alive).sum(1) / np.maximum(n, 1), 0.0)
    add("hp_frac_diff_alive", mean_alive_hp(slice(0, 5)) - mean_alive_hp(slice(5, 10)))

    # movement/tempo: the transformer's micro-deltas, summed per side
    micro = bundle["champ"].astype(np.float32)[:, :, OFF_MICRO:OFF_MICRO + 12]
    for j, mname in enumerate(MICRO_NAMES):
        add(f"micro_diff_{mname}", micro[:, :5, j].sum(1) - micro[:, 5:, j].sum(1))
    for r, role in enumerate(("top", "jungle", "mid", "bot", "sup")):
        add(f"micro_cs30_diff_{role}", micro[:, r, 0] - micro[:, 5 + r, 0])
    add("micro_disp30_blue", micro[:, :5, 10].mean(1))
    add("micro_disp30_red", micro[:, 5:, 10].mean(1))

    return np.stack(cols, axis=1), names


def write_preds(out_root: str | Path, name: str, bundles_dir: str | Path,
                config: dict, preds_by_split: dict[str, dict]) -> Path:
    """Project-wide predictions contract: <out>/<name>/<split>_preds.npz + meta.json.

    preds_by_split: {split: {game: float32 [T] p(blue wins)}}.
    """
    out = Path(out_root) / name
    out.mkdir(parents=True, exist_ok=True)
    for split, preds in preds_by_split.items():
        np.savez(out / f"{split}_preds.npz",
                 **{g: np.asarray(p, dtype=np.float32) for g, p in preds.items()})
    meta = {
        "model": name,
        "split_source": str(bundles_dir),
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "config": config,
    }
    with open(out / "meta.json", "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
    return out
