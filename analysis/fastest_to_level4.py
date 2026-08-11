"""Find, per jungle champion, the game with the fastest time to level 4.

Reads the dataset in any of its forms:
  - a processed dir  (data/processed/<game>/{meta.json, sequence.npz})   [default]
  - a pack dir       (has shards/*.tar + features.json)                   --source data/pack
  - a Hugging Face dataset repo (downloads a snapshot first)              --hf zauberine/lol-winprob

For each participant whose role is Jungle, time-to-level-4 is the first per-second
timestamp `t` at which that player's raw `level` feature reaches 4. We keep the
minimum such time per champion, and report which game/player/patch it came from.

By default we EXCLUDE clears where the jungler cast Smite 2+ times before level 4
(a double-Smite clear is a different route, not comparable to a standard clear).
Smite casts are detected as rising edges in the summoner-spell cooldown feature
for whichever summoner slot holds Smite (spell id 11). Tune with --max-smite.

Usage:
  python analysis/fastest_to_level4.py
  python analysis/fastest_to_level4.py --source data/pack --csv l4.csv
  python analysis/fastest_to_level4.py --hf zauberine/lol-winprob
  python analysis/fastest_to_level4.py --max-smite 99      # disable the filter
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
SMITE_SPELL_ID = 11          # Riot summoner-spell id for Smite
SMITE_CAST_JUMP = 5.0        # a per-second rise this large in the summ-cd = a cast
                             # (base decay is ~-1/s; a Smite cast jumps cd to ~15+)


def champ_feature_indices(root: Path) -> dict[str, int]:
    feats = DEFAULT_CHAMP_FEATURES
    fj = root / "features.json"
    if fj.exists():
        j = json.loads(fj.read_text(encoding="utf-8"))
        if isinstance(j.get("X_champ"), list):
            feats = j["X_champ"]
    return {name: feats.index(name) for name in ("level", "cd_summ1", "cd_summ2")}


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


def smite_casts_before(cd_series: np.ndarray, l4_idx: int) -> int:
    """Count Smite casts strictly before the level-4 frame, as rising edges in
    the summoner-cooldown series (a cast jumps cd up; otherwise it decays)."""
    pre = cd_series[:l4_idx]
    if pre.size < 2:
        return 0
    return int(np.count_nonzero(np.diff(pre) > SMITE_CAST_JUMP))


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
    ap.add_argument("--max-smite", type=int, default=1,
                    help="exclude junglers with MORE than this many Smite casts "
                         "before level 4 (default 1, i.e. drop double-Smite clears; "
                         "use a big number to disable)")
    args = ap.parse_args()

    if args.hf:
        from huggingface_hub import snapshot_download
        root = Path(snapshot_download(args.hf, repo_type="dataset"))
    else:
        root = Path(args.source)
    if not root.exists():
        raise SystemExit(f"source not found: {root}")

    idx = champ_feature_indices(root)
    LVL, S1, S2 = idx["level"], idx["cd_summ1"], idx["cd_summ2"]
    best: dict[str, dict] = {}
    n_games = n_junglers = n_excluded = 0

    for game, meta, npz in iter_games(root):
        n_games += 1
        t = npz["t"]
        X = npz["X_champ"]
        for i, p in enumerate(meta.get("participants", [])):
            if str(p.get("role", "")).lower() != "jungle":
                continue
            n_junglers += 1
            reached = X[:, i, LVL] >= TARGET_LEVEL
            if not reached.any():
                continue
            l4 = int(np.argmax(reached))
            secs = int(t[l4])

            spells = p.get("summoner_spells", []) or []
            smite_slot = spells.index(SMITE_SPELL_ID) if SMITE_SPELL_ID in spells else None
            n_smite = (smite_casts_before(X[:, i, S1 if smite_slot == 0 else S2], l4)
                       if smite_slot is not None else 0)
            if n_smite > args.max_smite:
                n_excluded += 1
                continue

            champ = p.get("champion", "?")
            cur = best.get(champ)
            if cur is None or secs < cur["seconds"]:
                best[champ] = {
                    "champion": champ, "seconds": secs, "smites_before_l4": n_smite,
                    "game": game, "player": p.get("player", ""),
                    "team": p.get("team", ""), "patch": meta.get("patch", ""),
                }

    rows = list(best.values())
    rows.sort(key=(lambda r: r["seconds"]) if args.sort == "time"
              else (lambda r: r["champion"]))

    print(f"scanned {n_games} games, {n_junglers} jungle appearances, "
          f"{len(rows)} distinct jungle champions "
          f"(excluded {n_excluded} clears with >{args.max_smite} Smite before lvl 4)\n")
    print(f"{'champion':<14} {'t->lvl4':>8}  {'m:ss':>6}  {'smite':>5}  "
          f"{'patch':<16} {'game':<16} player")
    print("-" * 96)
    for r in rows:
        s = r["seconds"]
        print(f"{r['champion']:<14} {s:>7}s  {s//60}:{s%60:02d}   {r['smites_before_l4']:>5}  "
              f"{str(r['patch']):<16} {r['game']:<16} {r['player']}")

    if args.csv:
        with open(args.csv, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=["champion", "seconds", "smites_before_l4",
                                               "game", "player", "team", "patch"])
            w.writeheader()
            w.writerows(rows)
        print(f"\nwrote {args.csv}")


if __name__ == "__main__":
    main()
