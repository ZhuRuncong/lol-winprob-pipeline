"""Deep ensemble: probability-space mean over members + exact test-time
symmetrization p* = (p_mean(x) + 1 - p_mean(swap(x))) / 2.

Reports the antisymmetry-violation metric mean |p(x) + p(swap(x)) - 1|
(how much work symmetrization is doing) as a standing diagnostic.

Usage:
  python -m pipeline.winprob.ensemble --members results/member_s1 results/member_s2 \
      --out results/ensemble --splits val test
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def load_preds(member: Path, split: str, swap: bool) -> dict[str, np.ndarray]:
    path = member / f"{split}_preds{'_swap' if swap else ''}.npz"
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--members", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--splits", nargs="+", default=["val", "test"])
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    report = {"members": args.members, "splits": {}}

    for split in args.splits:
        mem_p = [load_preds(Path(m), split, swap=False) for m in args.members]
        mem_ps = [load_preds(Path(m), split, swap=True) for m in args.members]
        games = sorted(mem_p[0])
        for mp in mem_p + mem_ps:
            if sorted(mp) != games:
                raise SystemExit(f"member game sets differ on {split}")

        sym, mean_p, mean_ps, viols = {}, {}, {}, []
        for g in games:
            p = np.mean([m[g] for m in mem_p], axis=0)
            ps = np.mean([m[g] for m in mem_ps], axis=0)
            viols.append(float(np.mean(np.abs(p + ps - 1.0))))
            sym[g] = ((p + 1.0 - ps) / 2.0).astype(np.float32)
            mean_p[g] = p.astype(np.float32)
            mean_ps[g] = ps.astype(np.float32)
        np.savez(out / f"{split}_preds.npz", **sym)
        # unsymmetrized streams: calibrate.py calibrates these and
        # symmetrizes LAST (the plan's order: mean -> calibrator -> symmetrize)
        np.savez(out / f"{split}_mean.npz", **mean_p)
        np.savez(out / f"{split}_mean_swap.npz", **mean_ps)
        report["splits"][split] = {
            "n_games": len(games),
            "antisymmetry_violation_mean": float(np.mean(viols)),
            "antisymmetry_violation_max": float(np.max(viols)),
        }
        print(f"{split}: {len(games)} games, K={len(mem_p)} members, "
              f"antisym violation mean={np.mean(viols):.4f}")

    report["created_utc"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    with open(out / "meta.json", "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=1)


if __name__ == "__main__":
    main()
