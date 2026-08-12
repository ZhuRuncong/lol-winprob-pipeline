"""Post-hoc calibration, fit on VAL only, applied ONCE to test.

Compares (i) global temperature, (ii) per-absolute-time-bucket temperature,
(iii) isotonic regression — all on per-game-weighted val seconds — and
selects by val per-game log-loss. Emits calibrated prediction files plus a
pre/post report. Probabilities in, probabilities out (works on any prediction
directory, ensemble or single model).

Usage:
  python -m pipeline.winprob.calibrate --preds results/ensemble \
      --bundles data/winprob_bundles --out results/ensemble_cal
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

BUCKETS_S = [(0, 600), (600, 900), (900, 1200), (1200, 1500), (1500, 1800), (1800, 10 ** 9)]
EPS = 1e-6


def load_preds(d: Path, split: str) -> dict[str, np.ndarray]:
    with np.load(d / f"{split}_preds.npz") as z:
        return {k: z[k] for k in z.files}


def labels_of(bundles: Path) -> dict[str, int]:
    with open(bundles / "bundles_manifest.json", encoding="utf-8") as fh:
        return {g["game"]: g["y"] for g in json.load(fh)["games"]}


def logits(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def per_game_logloss(preds: dict, ys: dict) -> float:
    lls = []
    for g, p in preds.items():
        p = np.clip(p, EPS, 1 - EPS)
        y = ys[g]
        lls.append(float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))))
    return float(np.mean(lls))


def fit_temperature(z_list: list[np.ndarray], y_list: list[float]) -> float:
    """1-D NLL minimization over T in [0.25, 4], per-game-weighted."""
    def nll(T):
        tot = 0.0
        for z, y in zip(z_list, y_list):
            p = 1 / (1 + np.exp(-z / T))
            p = np.clip(p, EPS, 1 - EPS)
            tot += -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
        return tot / max(len(z_list), 1)
    ts = np.exp(np.linspace(np.log(0.25), np.log(4.0), 60))
    coarse = ts[int(np.argmin([nll(t) for t in ts]))]
    fine = np.linspace(coarse * 0.85, coarse * 1.18, 40)
    return float(fine[int(np.argmin([nll(t) for t in fine]))])


def apply_global(preds, T):
    return {g: 1 / (1 + np.exp(-logits(p) / T)) for g, p in preds.items()}


def apply_bucketed(preds, temps):
    out = {}
    for g, p in preds.items():
        z = logits(p)
        q = np.empty_like(p)
        for (a, b), T in zip(BUCKETS_S, temps):
            sl = slice(a, min(b, len(p)))
            q[sl] = 1 / (1 + np.exp(-z[sl] / T))
        out[g] = q.astype(np.float32)
    return out


def fit_calibrators(val: dict, ys: dict):
    """Fit all candidate calibrators on the val stream; return name->fn."""
    zg = [logits(p) for p in val.values()]
    yg = [float(ys[g]) for g in val]
    T_global = fit_temperature(zg, yg)

    temps = []
    for a, b in BUCKETS_S:
        zs, yl = [], []
        for g, p in val.items():
            sl = slice(a, min(b, len(p)))
            if sl.stop > sl.start:
                zs.append(logits(p[sl]))
                yl.append(float(ys[g]))
        temps.append(fit_temperature(zs, yl) if zs else 1.0)

    from sklearn.isotonic import IsotonicRegression
    ps = np.concatenate([p for p in val.values()])
    yy = np.concatenate([np.full(len(p), float(ys[g])) for g, p in val.items()])
    ww = np.concatenate([np.full(len(p), 1.0 / len(p)) for p in val.values()])
    iso = IsotonicRegression(y_min=EPS, y_max=1 - EPS, out_of_bounds="clip")
    iso.fit(ps, yy, sample_weight=ww)

    fns = {
        "identity": lambda pr: pr,
        "temp_global": lambda pr: apply_global(pr, T_global),
        "temp_bucketed": lambda pr: apply_bucketed(pr, temps),
        "isotonic": lambda pr: {g: iso.predict(p).astype(np.float32)
                                for g, p in pr.items()},
    }
    params = {"T_global": T_global,
              "T_bucketed": dict(zip([f"{a}-{b}s" for a, b in BUCKETS_S], temps))}
    return fns, params


def symmetrize(p: dict, p_swap: dict) -> dict:
    return {g: ((p[g] + 1.0 - p_swap[g]) / 2.0).astype(np.float32) for g in p}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preds", required=True)
    ap.add_argument("--bundles", default="data/winprob_bundles")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    src, out, bundles = Path(args.preds), Path(args.out), Path(args.bundles)
    out.mkdir(parents=True, exist_ok=True)
    ys = labels_of(bundles)

    # When ensemble.py's unsymmetrized streams exist, follow the plan's order:
    # mean -> calibrator (fit on the val mean stream, applied to BOTH the mean
    # and swap streams) -> symmetrize LAST. Otherwise calibrate a plain
    # prediction dir directly (baselines).
    has_swap = (src / "val_mean.npz").exists()

    def load_stream(split, name):
        with np.load(src / f"{split}_{name}.npz") as z:
            return {k: z[k] for k in z.files}

    if has_swap:
        val_m, val_s = load_stream("val", "mean"), load_stream("val", "mean_swap")
        test_m, test_s = load_stream("test", "mean"), load_stream("test", "mean_swap")
        fns, params = fit_calibrators(val_m, ys)
        finals_val = {n: symmetrize(fn(val_m), fn(val_s)) for n, fn in fns.items()}
        finals_test = {n: symmetrize(fn(test_m), fn(test_s)) for n, fn in fns.items()}
        pre_val, pre_test = symmetrize(val_m, val_s), symmetrize(test_m, test_s)
    else:
        pre_val, pre_test = load_preds(src, "val"), load_preds(src, "test")
        fns, params = fit_calibrators(pre_val, ys)
        finals_val = {n: fn(pre_val) for n, fn in fns.items()}
        finals_test = {n: fn(pre_test) for n, fn in fns.items()}

    # selection by val per-game log-loss of the FINAL (shipped-form) output
    val_ll = {n: per_game_logloss(p, ys) for n, p in finals_val.items()}
    winner = min(val_ll, key=val_ll.get)
    np.savez(out / "val_preds.npz", **finals_val[winner])
    np.savez(out / "test_preds.npz", **finals_test[winner])

    report = {
        "selected": winner,
        "symmetrize_last": has_swap,
        "val_per_game_logloss": val_ll,
        **params,
        "test_logloss_pre": per_game_logloss(pre_test, ys),
        "test_logloss_post": per_game_logloss(finals_test[winner], ys),
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source": str(src),
    }
    with open(out / "calibration_report.json", "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=1)
    print(f"selected: {winner}  (symmetrize_last={has_swap})  val log-loss: " +
          "  ".join(f"{k}={v:.5f}" for k, v in val_ll.items()))
    print(f"test log-loss pre={report['test_logloss_pre']:.5f} "
          f"post={report['test_logloss_post']:.5f}")


if __name__ == "__main__":
    main()
