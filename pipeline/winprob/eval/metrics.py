"""Metric library for win-probability evaluation.

Games are the atomic unit everywhere: every metric averages within a game
first and then macro over games; pooled quantities (ECE) weight each second
1/T_g so every game counts equally; the bootstrap resamples SERIES, never
games or seconds. Probabilities are clipped to [1e-6, 1-1e-6] only inside
the log-loss computation.

`preds` is always {game: float32 [T] p(blue wins)}, `ys` is {game: 0/1}.
"""
from __future__ import annotations

import numpy as np

CLIP = 1e-6

# Absolute-time buckets (upper edges in seconds; the last bucket is open).
ABS_EDGES = np.array([600, 900, 1200, 1500, 1800])
ABS_LABELS = ["0-10", "10-15", "15-20", "20-25", "25-30", "30+"]
N_ABS = len(ABS_LABELS)
FRAC_LABELS = ["q1", "q2", "q3", "q4"]


def abs_bucket_index(t_seconds: np.ndarray) -> np.ndarray:
    return np.searchsorted(ABS_EDGES, t_seconds, side="right")


def frac_bucket_index(T: int) -> np.ndarray:
    return np.minimum(np.arange(T) * 4 // T, 3)


# ---------------------------------------------------------------------------
# Per-game metrics (mean over t within one game; None when the mask is empty)
# ---------------------------------------------------------------------------

def game_logloss(p: np.ndarray, y: int, mask: np.ndarray | None = None) -> float | None:
    pc = np.clip(np.asarray(p, dtype=np.float64), CLIP, 1.0 - CLIP)
    ll = -np.log(pc) if y == 1 else -np.log(1.0 - pc)
    if mask is not None:
        ll = ll[mask]
    return float(ll.mean()) if ll.size else None


def game_brier(p: np.ndarray, y: int, mask: np.ndarray | None = None) -> float | None:
    e = (np.asarray(p, dtype=np.float64) - y) ** 2
    if mask is not None:
        e = e[mask]
    return float(e.mean()) if e.size else None


def game_accuracy(p: np.ndarray, y: int, mask: np.ndarray | None = None) -> float | None:
    hit = ((np.asarray(p) > 0.5) == bool(y)).astype(np.float64)
    if mask is not None:
        hit = hit[mask]
    return float(hit.mean()) if hit.size else None


# ---------------------------------------------------------------------------
# Macro aggregates
# ---------------------------------------------------------------------------

def per_game_logloss(preds: dict, ys: dict) -> float:
    return float(np.mean([game_logloss(preds[g], ys[g]) for g in preds]))


def per_game_brier(preds: dict, ys: dict) -> float:
    return float(np.mean([game_brier(preds[g], ys[g]) for g in preds]))


def logloss_by_game(preds: dict, ys: dict,
                    lo_s: float | None = None, hi_s: float | None = None) -> dict:
    """{game: per-game log-loss}, optionally restricted to seconds [lo_s, hi_s).

    Games with no seconds in the window are omitted.
    """
    out = {}
    for g, p in preds.items():
        mask = None
        if lo_s is not None or hi_s is not None:
            ts = np.arange(len(p))
            mask = (ts >= (lo_s or 0)) & (ts < (hi_s if hi_s is not None else np.inf))
        v = game_logloss(p, ys[g], mask)
        if v is not None:
            out[g] = v
    return out


# ---------------------------------------------------------------------------
# Dual bucketing
# ---------------------------------------------------------------------------

def bucket_table(preds: dict, ys: dict, kind: str = "abs") -> list[dict]:
    """Per-bucket per-game-macro metrics.

    kind "abs": absolute-time buckets; a game contributes to a bucket only
    where it has seconds there. kind "frac": fraction-of-game quartiles.
    """
    labels = ABS_LABELS if kind == "abs" else FRAC_LABELS
    rows = []
    for b, lab in enumerate(labels):
        lls, brs, accs = [], [], []
        for g, p in preds.items():
            idx = abs_bucket_index(np.arange(len(p))) if kind == "abs" \
                else frac_bucket_index(len(p))
            mask = idx == b
            ll = game_logloss(p, ys[g], mask)
            if ll is None:
                continue
            lls.append(ll)
            brs.append(game_brier(p, ys[g], mask))
            accs.append(game_accuracy(p, ys[g], mask))
        rows.append({
            "bucket": lab, "n_games": len(lls),
            "logloss": float(np.mean(lls)) if lls else None,
            "brier": float(np.mean(brs)) if brs else None,
            "accuracy": float(np.mean(accs)) if accs else None,
        })
    return rows


def accuracy_vs_time(preds: dict, ys: dict) -> list[dict]:
    """Accuracy per absolute-time bucket (per-game macro)."""
    return [{"bucket": r["bucket"], "n_games": r["n_games"], "accuracy": r["accuracy"]}
            for r in bucket_table(preds, ys, kind="abs")]


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

def ece(preds: dict, ys: dict, n_bins: int = 15) -> tuple[float, list[dict]]:
    """Expected calibration error over pooled per-second predictions.

    15 equal-mass bins; each second carries weight 1/T_g so every game
    contributes equally. Returns (ece, reliability curve points).
    """
    p = np.concatenate([np.asarray(preds[g], dtype=np.float64) for g in preds])
    y = np.concatenate([np.full(len(preds[g]), ys[g], dtype=np.float64) for g in preds])
    w = np.concatenate([np.full(len(preds[g]), 1.0 / len(preds[g])) for g in preds])

    order = np.argsort(p, kind="stable")
    p, y, w = p[order], y[order], w[order]
    total = w.sum()
    # bin by the cumulative-mass midpoint of each sample
    cum = np.cumsum(w) - 0.5 * w
    idx = np.minimum((cum / total * n_bins).astype(np.int64), n_bins - 1)

    value, points = 0.0, []
    for b in range(n_bins):
        m = idx == b
        wb = w[m].sum()
        if wb <= 0:
            continue
        conf = float((p[m] * w[m]).sum() / wb)
        acc = float((y[m] * w[m]).sum() / wb)
        value += (wb / total) * abs(acc - conf)
        points.append({"p_mean": conf, "y_mean": acc, "weight": float(wb / total)})
    return float(value), points


# ---------------------------------------------------------------------------
# Paired series-level bootstrap (the decision rule)
# ---------------------------------------------------------------------------

def paired_bootstrap(ll_a: dict, ll_b: dict, series_by_game: dict,
                     n: int = 1000, seed: int = 0) -> dict:
    """95% CI on mean per-game delta (A - B) by resampling SERIES.

    ll_a / ll_b are {game: per-game log-loss}; only common games count.
    Negative delta = A better.
    """
    games = sorted(set(ll_a) & set(ll_b))
    delta = {g: ll_a[g] - ll_b[g] for g in games}
    by_series: dict[str, list[float]] = {}
    for g in games:
        by_series.setdefault(series_by_game[g], []).append(delta[g])
    series = sorted(by_series)

    rng = np.random.default_rng(seed)
    means = np.empty(n)
    for i in range(n):
        picked = rng.integers(0, len(series), size=len(series))
        vals = [v for j in picked for v in by_series[series[j]]]
        means[i] = np.mean(vals)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return {
        "mean_delta": float(np.mean(list(delta.values()))),
        "ci_lo": float(lo), "ci_hi": float(hi),
        "n_games": len(games), "n_series": len(series), "n_boot": n,
        "excludes_zero": ci_excludes_zero(float(lo), float(hi)),
    }


def ci_excludes_zero(lo: float, hi: float) -> bool:
    return lo > 0.0 or hi < 0.0
