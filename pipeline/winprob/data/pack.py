"""HF shards -> per-game training bundles (canonical role order, normalized).

Each bundle is one uncompressed .npz under <out>/<split>/<game>.npz:
  champ    f16 [T,10,58]  z-applied continuous champion block
  team     f16 [T,2,46]   z-applied team block (incl. ward vision features)
  glob     f16 [T,12]     global block (transformed + clock Fourier)
  items    i16 [T,10,7]   hash-bucket item indices (0 = empty)
  item_cd  f16 [T,10,7]   bounded item cooldowns
  statics  i16 [10,7]     champion_id, role, spells, keystone, styles
  tgt_champ        f32 [T,10,7]   raw SSL target values (clean, un-augmented)
  tgt_champ_deaths f32 [T,10]
  tgt_team         f32 [T,2,5]
  pos_bin          i16 [T,10]
  y i8, patch_idx i16
Champion axis is ALWAYS canonical role order: [blue Top,Jg,Mid,Bot,Sup,
red Top,Jg,Mid,Bot,Sup]. A bundles manifest records provenance
(norm_version, revision, git SHA) and per-game rows.

Usage:
  python -m pipeline.winprob.data.pack [--local-sample] [--out data/winprob_bundles]
"""
from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from ..config import HF_REVISION
from . import transforms as tr
from .hub import corpus_root, iter_shard_games
from .splits import load_manifest, temporal_split_with_val
from .stats import load_stats


def git_sha() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, timeout=10,
                              cwd=Path(__file__).parent).stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def pack_game(meta: dict, npz: dict, stats: dict) -> tuple[dict[str, np.ndarray], bool]:
    perm, roles_ok = tr.role_order(meta.get("participants", []))

    X_champ = npz["X_champ"][:, perm]
    X_kda = npz["X_champ_kda"][:, perm]
    X_items = npz["X_items"][:, perm]
    X_item_cd = npz["X_item_cd"][:, perm]
    X_team, X_global, wards = npz["X_team"], npz["X_global"], npz["wards"]
    T = X_champ.shape[0]

    champ = tr.champ_cont_prenorm(X_champ, X_kda)
    tr.apply_z(champ, tr.CHAMP_Z_MASK,
               np.asarray(stats["champ_mean"]), np.asarray(stats["champ_std"]))
    team = tr.team_cont_prenorm(X_team, wards, T)
    tr.apply_z(team, tr.TEAM_Z_MASK,
               np.asarray(stats["team_mean"]), np.asarray(stats["team_std"]))

    participants = [meta["participants"][i] for i in perm]
    targets = tr.build_targets(X_champ, X_kda, X_team)

    patch_idx = stats["patch_vocab"].get(tr.parse_patch(meta.get("patch")), 0)

    bundle = {
        "champ": champ.astype(np.float16),
        "team": team.astype(np.float16),
        "glob": tr.global_cont(X_global, T).astype(np.float16),
        "items": tr.hash_items(X_items),
        "item_cd": tr.item_cd_feats(X_item_cd).astype(np.float16),
        "statics": tr.champ_statics(participants),
        "tgt_champ": targets["tgt_champ"],
        "tgt_champ_deaths": targets["tgt_champ_deaths"],
        "tgt_team": targets["tgt_team"],
        "pos_bin": targets["pos_bin"],
        "y": np.int8(meta["label"]),
        "patch_idx": np.int16(patch_idx),
    }
    return bundle, roles_ok


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--local-sample", action="store_true")
    ap.add_argument("--split", choices=["manifest", "temporal"], default="manifest",
                    help="'temporal' packs the late-patch-holdout splits (stats must "
                         "have been fit with the same --split)")
    ap.add_argument("--out", default="data/winprob_bundles")
    ap.add_argument("--stats", default=None,
                    help="norm_v2.json path (default <out>/norm_v2.json)")
    args = ap.parse_args()

    out = Path(args.out)
    stats = load_stats(args.stats or out / "norm_v2.json")
    if stats.get("split_source", "manifest") != args.split:
        raise SystemExit(f"stats were fit with split_source="
                         f"{stats.get('split_source')!r}, pack asked for {args.split!r}")

    root = corpus_root(local_sample=args.local_sample)
    manifest = load_manifest(root)
    if args.split == "temporal":
        assign = temporal_split_with_val(manifest)
        rows_by_game = {g["game"]: {**g, "split": s}
                        for s, rs in assign.items() for g in rs}
    else:
        rows_by_game = {g["game"]: g for g in manifest["games"]}

    packed_rows, n_bad_roles = [], 0
    for i, (game, meta, npz) in enumerate(iter_shard_games(root), 1):
        row = rows_by_game.get(game)
        if row is None:
            continue
        if row.get("label") is not None and row["label"] != meta["label"]:
            raise SystemExit(f"{game}: manifest label {row['label']} != meta label {meta['label']}")
        bundle, roles_ok = pack_game(meta, npz, stats)
        n_bad_roles += (not roles_ok)

        dest = out / row["split"] / f"{game}.npz"
        dest.parent.mkdir(parents=True, exist_ok=True)
        np.savez(dest, **bundle)

        packed_rows.append({
            "game": game, "series_id": row["series_id"], "split": row["split"],
            "T": int(npz["X_champ"].shape[0]), "y": int(meta["label"]),
            "patch": meta.get("patch"), "patch_idx": int(bundle["patch_idx"]),
            "roles_ok": bool(roles_ok),
        })
        if i % 250 == 0:
            print(f"  packed {i} games", flush=True)

    bm = {
        "norm_version": stats["norm_version"],
        "split_source": args.split,
        "hf_revision": HF_REVISION if not args.local_sample else "local-sample",
        "git_sha": git_sha(),
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "n_games": len(packed_rows),
        "games": packed_rows,
    }
    with open(out / "bundles_manifest.json", "w", encoding="utf-8") as fh:
        json.dump(bm, fh)
    counts = {s: sum(1 for r in packed_rows if r["split"] == s)
              for s in ("train", "val", "test")}
    print(f"packed {len(packed_rows)} games -> {out}  splits={counts}  "
          f"role-fallback games={n_bad_roles}  norm={stats['norm_version']}")


if __name__ == "__main__":
    main()
