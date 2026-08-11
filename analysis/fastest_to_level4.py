"""Find, per jungle champion, the game with the fastest time to level 4.

Reads the dataset in any of its forms:
  - a processed dir  (data/processed/<game>/{meta.json, sequence.npz})   [default]
  - a pack dir       (has shards/*.tar + features.json)                   --source data/pack
  - a Hugging Face dataset repo (downloads a snapshot first)              --hf zauberine/lol-winprob

For each participant whose role is Jungle, time-to-level-4 is the first per-second
timestamp `t` at which that player's raw `level` feature reaches 4. We keep the
minimum such time per champion, and report which game/player/patch it came from.

Usage:
  python analysis/fastest_to_level4.py
  python analysis/fastest_to_level4.py --source data/pack --csv l4.csv --sort time
  python analysis/fastest_to_level4.py --hf zauberine/lol-winprob
"""
from __future__ import annotations
import argparse
import csv
import io
import json
import tarfile
from pathlib import Path

import numpy as np

# fallback feature order (pipeline/build_processed.py FEATURES_CHAMP) if the
# dataset has no features.json alongside it.
DEFAULT_CHAMP_FEATURES = [
    "alive", "respawn_timer", "level", "xp", "current_gold", "total_gold",
    "hp", "hp_max", "hp_frac", "mana", "mana_max", "mana_frac",
    "cd_q", "cd_w", "cd_e", "cd_r", "lvl_q", "lvl_w", "lvl_e", "lvl_r",
    "cd_summ1", "cd_summ2", "pos_x", "pos_z", "shutdown_value", "cs",
]
TARGET_LEVEL = 4


def level_index(root: Path) -> int:
    fj = root / "features.json"
    if fj.exists():
        feats = json.loads(fj.read_text(encoding="utf-8"))
        if "X_champ" in feats and "level" in feats["X_champ"]:
            return feats["X_champ"].index("level")
    return DEFAULT_CHAMP_FEATURES.index("level")


def iter_games(root: Path):
    """Yield (game_name, meta_dict, npz) from a pack dir or a processed dir."""
    if (root / "shards").is_dir():
        for tar_path in sorted((root / "shards").glob("*.tar")):
            with tarfile.open(tar_path) as tf:
                groups: dict[str, dict[str, tarfile.TarInfo]] = {}
                for m in tf.getmembers():
                    if not m.isfile():
                        continue
                    parts = m.name.split("/")
                    if len(parts) >= 2:
                        groups.setdefault(parts[0], {})[parts[-1]] = m
                for game, files in groups.items():
                    if "meta.json" in files and "sequence.npz" in files:
                        meta = json.loads(tf.extractfile(files["meta.json"]).read().decode("utf-8"))
                        npz = np.load(io.BytesIO(tf.extractfile(files["sequence.npz"]).read()))
                        yield game, meta, npz
    else:
        for d in sorted(p for p in root.iterdir() if p.is_dir()):
            if (d / "meta.json").exists() and (d / "sequence.npz").exists():
                meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
                yield d.name, meta, np.load(d / "sequence.npz")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default="data/processed",
                    help="processed dir or pack dir (default data/processed)")
    ap.add_argument("--hf", metavar="REPO",
                    help="download this HF dataset repo and read it instead")
    ap.add_argument("--csv", help="also write results to this CSV path")
    ap.add_argument("--sort", choices=["time", "champion"], default="time",
                    help="order the output by fastest time (default) or champion name")
    args = ap.parse_args()

    if args.hf:
        from huggingface_hub import snapshot_download
        root = Path(snapshot_download(args.hf, repo_type="dataset"))
    else:
        root = Path(args.source)
    if not root.exists():
        raise SystemExit(f"source not found: {root}")

    lvl = level_index(root)
    best: dict[str, dict] = {}
    n_games = n_junglers = 0

    for game, meta, npz in iter_games(root):
        n_games += 1
        t = npz["t"]
        X = npz["X_champ"]
        for i, p in enumerate(meta.get("participants", [])):
            if str(p.get("role", "")).lower() != "jungle":
                continue
            n_junglers += 1
            reached = X[:, i, lvl] >= TARGET_LEVEL
            if not reached.any():
                continue
            secs = int(t[int(np.argmax(reached))])
            champ = p.get("champion", "?")
            cur = best.get(champ)
            if cur is None or secs < cur["seconds"]:
                best[champ] = {
                    "champion": champ, "seconds": secs, "game": game,
                    "player": p.get("player", ""), "team": p.get("team", ""),
                    "patch": meta.get("patch", ""),
                }

    rows = list(best.values())
    rows.sort(key=(lambda r: r["seconds"]) if args.sort == "time"
              else (lambda r: r["champion"]))

    print(f"scanned {n_games} games, {n_junglers} jungle appearances, "
          f"{len(rows)} distinct jungle champions\n")
    print(f"{'champion':<14} {'t->lvl4':>8}  {'m:ss':>6}  {'patch':<16} {'game':<16} player")
    print("-" * 88)
    for r in rows:
        s = r["seconds"]
        print(f"{r['champion']:<14} {s:>7}s  {s//60}:{s%60:02d}   "
              f"{str(r['patch']):<16} {r['game']:<16} {r['player']}")

    if args.csv:
        with open(args.csv, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=["champion", "seconds", "game",
                                               "player", "team", "patch"])
            w.writeheader()
            w.writerows(rows)
        print(f"\nwrote {args.csv}")


if __name__ == "__main__":
    main()
