"""Evaluation report over a predictions directory.

Prints overall + dual-bucketed per-game metrics and ECE, writes report.json
next to the predictions. With --compare, adds the paired series-level
bootstrap table (delta per-game log-loss, 95% CI) including the 10-25 min
decision window.

Usage:
  python -m pipeline.winprob.eval.report --bundles data/winprob_bundles_local \
      --preds data/winprob_results_local/gbt \
      [--compare data/winprob_results_local/logistic] [--split test]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .metrics import (ABS_LABELS, accuracy_vs_time, bucket_table, ece,
                      logloss_by_game, paired_bootstrap, per_game_brier,
                      per_game_logloss)

# decision window: the 10-25 min buckets
DECISION_LO, DECISION_HI = 600, 1500


def load_split_rows(bundles_dir: str | Path, split: str) -> list[dict]:
    with open(Path(bundles_dir) / "bundles_manifest.json", encoding="utf-8") as fh:
        manifest = json.load(fh)
    rows = [g for g in manifest["games"] if g["split"] == split]
    if not rows:
        raise SystemExit(f"no games for split '{split}' in {bundles_dir}")
    return rows


def load_preds(preds_dir: str | Path, split: str, rows: list[dict]) -> dict:
    path = Path(preds_dir) / f"{split}_preds.npz"
    with np.load(path) as z:
        preds = {g: z[g] for g in z.files}
    missing = [r["game"] for r in rows if r["game"] not in preds]
    if missing:
        raise SystemExit(f"{path}: missing {len(missing)} games, e.g. {missing[:3]}")
    for r in rows:
        p = preds[r["game"]]
        if len(p) != r["T"]:
            raise SystemExit(f"{path}: {r['game']} has {len(p)} preds, manifest T={r['T']}")
        if not np.all(np.isfinite(p)):
            raise SystemExit(f"{path}: {r['game']} has non-finite predictions")
    return {r["game"]: preds[r["game"]] for r in rows}


def print_buckets(title: str, table: list[dict]):
    print(f"\n{title}")
    print(f"  {'bucket':>7} {'games':>5} {'logloss':>8} {'brier':>8} {'acc':>7}")
    for r in table:
        if r["logloss"] is None:
            print(f"  {r['bucket']:>7} {r['n_games']:>5}        -        -       -")
        else:
            print(f"  {r['bucket']:>7} {r['n_games']:>5} {r['logloss']:>8.4f} "
                  f"{r['brier']:>8.4f} {r['accuracy']:>7.3f}")


def compare_table(preds_a: dict, preds_b: dict, ys: dict, series: dict) -> list[dict]:
    """Paired bootstrap rows: overall, each abs bucket, and the decision window."""
    windows = [("overall", None, None)]
    edges = [0, 600, 900, 1200, 1500, 1800, None]
    windows += [(lab, edges[i], edges[i + 1]) for i, lab in enumerate(ABS_LABELS)]
    windows.append(("10-25 (decision)", DECISION_LO, DECISION_HI))

    out = []
    for lab, lo, hi in windows:
        ll_a = logloss_by_game(preds_a, ys, lo, hi)
        ll_b = logloss_by_game(preds_b, ys, lo, hi)
        boot = paired_bootstrap(ll_a, ll_b, series, n=1000, seed=0)
        out.append({"window": lab, **boot})
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bundles", default="data/winprob_bundles_local")
    ap.add_argument("--preds", required=True)
    ap.add_argument("--compare", default=None)
    ap.add_argument("--split", default="test")
    args = ap.parse_args()

    rows = load_split_rows(args.bundles, args.split)
    preds = load_preds(args.preds, args.split, rows)
    ys = {r["game"]: r["y"] for r in rows}
    series = {r["game"]: r["series_id"] for r in rows}
    name = Path(args.preds).name

    report = {
        "model": name, "split": args.split, "bundles": str(args.bundles),
        "n_games": len(rows),
        "overall": {"logloss": per_game_logloss(preds, ys),
                    "brier": per_game_brier(preds, ys)},
        "abs_buckets": bucket_table(preds, ys, kind="abs"),
        "frac_quartiles": bucket_table(preds, ys, kind="frac"),
        "accuracy_vs_time": accuracy_vs_time(preds, ys),
    }
    report["ece"], report["reliability"] = ece(preds, ys)

    print(f"== {name} on {args.split} ({len(rows)} games) ==")
    print(f"per-game logloss {report['overall']['logloss']:.4f}   "
          f"brier {report['overall']['brier']:.4f}   ece {report['ece']:.4f}")
    print_buckets("absolute-time buckets (min)", report["abs_buckets"])
    print_buckets("fraction-of-game quartiles", report["frac_quartiles"])

    if args.compare:
        other = Path(args.compare).name
        preds_b = load_preds(args.compare, args.split, rows)
        table = compare_table(preds, preds_b, ys, series)
        report["compare"] = {"against": other, "windows": table}
        print(f"\npaired series bootstrap: delta logloss = {name} - {other} "
              f"(negative = {name} better)")
        print(f"  {'window':>17} {'games':>5} {'delta':>8} {'95% CI':>19} {'sig':>4}")
        for r in table:
            ci = f"[{r['ci_lo']:+.4f},{r['ci_hi']:+.4f}]"
            sig = "yes" if r["excludes_zero"] else "no"
            print(f"  {r['window']:>17} {r['n_games']:>5} {r['mean_delta']:>+8.4f} "
                  f"{ci:>19} {sig:>4}")

    out_path = Path(args.preds) / "report.json"
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
