"""The latent-state hypothesis harness.

Three tests, all run against a LABEL-FREE phase-A checkpoint (and compared
against a no-SSL trunk and raw features where applicable):

  horizon    held-out-horizon linear probes: 90s-ahead per-lane cs delta
             (90s is deliberately NOT in DELTA_HORIZONS) and
             next-objective-within-90s identity, probed from frozen h_t vs
             the same probe on raw per-second input features. The hypothesis
             requires SSL-representation probes to beat raw features.
  fog        masked-jungler localization: hide a jungler's dynamic inputs for
             60s spans, read the recon position head, report top-1/top-5 vs a
             last-seen persistence baseline. Beating persistence = inferred,
             not remembered, position.
  event-gain per-second log-loss delta vs a reference predictor (the GBDT),
             aligned to dragon/baron/tower takes. The hypothesis predicts the
             advantage concentrates 30-90s BEFORE the take.

Usage:
  python -m pipeline.winprob.probes horizon --ckpt runs/ssl_M/best.pt \
      [--ref-ckpt runs/scratch/best.pt] --bundles data/winprob_bundles
  python -m pipeline.winprob.probes fog --ckpt runs/ssl_M/best.pt --bundles ...
  python -m pipeline.winprob.probes event-gain --preds results/ensemble_cal \
      --ref-preds data/winprob_results/gbt --bundles ...
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .config import MaskConfig
from .data.dataset import GameBundleDataset
from .predict import load_model
from .train_common import autocast_ctx, pick_device
from .train_ssl import make_loader, to_device

HELD_OUT_H = 90            # not in DELTA_HORIZONS (5,30,120) — probe-only
LANE_SLOTS = [(0, 5), (2, 7), (3, 8)]   # top/mid/bot lane matchup slot pairs
JUNGLE_SLOTS = [1, 6]      # canonical role order: blue/red junglers


# ---------------------------------------------------------------- features

@torch.no_grad()
def collect(model_or_none, loader, ds_rows, cfg, device, stride=30):
    """Per game: (h_t or None, raw flat inputs, targets dict), subsampled."""
    rows = []
    for batch in loader:
        batch = to_device(batch, device)
        h = None
        if model_or_none is not None:
            with autocast_ctx(device, cfg.train.amp_dtype):
                h = model_or_none.trunk_features(batch).float().cpu().numpy()
        B, Tm = batch["valid"].shape
        champ = batch["champ"].float().cpu().numpy()
        team = batch["team"].float().cpu().numpy()
        glob = batch["glob"].float().cpu().numpy()
        raw = np.concatenate([champ.reshape(B, Tm, -1), team.reshape(B, Tm, -1), glob], -1)
        tgt_champ = batch["tgt_champ"].cpu().numpy()
        tgt_team = batch["tgt_team"].cpu().numpy()
        for b in range(B):
            T = int(batch["valid"][b].sum())
            hi = T - HELD_OUT_H
            if hi <= stride:
                continue
            idx = np.arange(0, hi, stride)
            # 90s-ahead cs delta for the three lane matchups (blue minus red)
            cs = tgt_champ[b, :, :, 0]
            lane_delta = np.stack(
                [(cs[idx + HELD_OUT_H, i] - cs[idx, i]) - (cs[idx + HELD_OUT_H, j] - cs[idx, j])
                 for i, j in LANE_SLOTS], 1)
            # next objective within 90s: 0 none, 1 blue dragon, 2 red dragon,
            # 3 blue tower, 4 red tower (first in priority order)
            drag = tgt_team[b, :, :, 2]
            tow = tgt_team[b, :, :, 1]
            obj = np.zeros(len(idx), dtype=np.int64)
            d0 = drag[idx + HELD_OUT_H] - drag[idx]
            t0 = tow[idx + HELD_OUT_H] - tow[idx]
            obj[t0[:, 1] >= 1] = 4
            obj[t0[:, 0] >= 1] = 3
            obj[d0[:, 1] >= 1] = 2
            obj[d0[:, 0] >= 1] = 1
            gi = int(batch["game_idx"][b])
            rows.append({
                "h": h[b, idx] if h is not None else None,
                "raw": raw[b, idx],
                "lane_delta": lane_delta,
                "obj": obj,
                "game": gi,
                "series": ds_rows[gi]["series_id"],
            })
    return rows


def probe_scores(rows, key) -> dict:
    """SERIES-grouped 2-fold ridge/logistic probe scores for one feature
    source (series is the project-wide grouping unit; games of one Bo5 must
    never straddle probe fit/eval)."""
    from sklearn.linear_model import LogisticRegression, Ridge
    series = sorted({r["series"] for r in rows})
    half = set(series[: len(series) // 2])
    tr = [r for r in rows if r["series"] in half]
    te = [r for r in rows if r["series"] not in half]

    def stack(rs, field):
        return np.concatenate([r[field] for r in rs])

    Xtr, Xte = stack(tr, key), stack(te, key)
    out = {}
    ytr, yte = stack(tr, "lane_delta"), stack(te, "lane_delta")
    r2 = []
    for lane in range(3):
        m = Ridge(alpha=10.0).fit(Xtr, ytr[:, lane])
        pred = m.predict(Xte)
        ss = 1 - np.sum((yte[:, lane] - pred) ** 2) / max(np.sum(
            (yte[:, lane] - yte[:, lane].mean()) ** 2), 1e-9)
        r2.append(float(ss))
    out["cs90_r2_mean"] = float(np.mean(r2))
    out["cs90_r2_lanes"] = r2

    otr, ote = stack(tr, "obj"), stack(te, "obj")
    clf = LogisticRegression(max_iter=2000, C=1.0).fit(Xtr, otr)
    out["obj90_acc"] = float((clf.predict(Xte) == ote).mean())
    out["obj90_base"] = float(np.bincount(ote, minlength=5).max() / len(ote))
    return out


def cmd_horizon(args):
    device = pick_device()
    model, cfg, _ = load_model(args.ckpt, device)
    ds = GameBundleDataset(args.bundles, "val")
    loader, _ = make_loader(ds, 65536, False, 0, 0)
    rows = collect(model, loader, ds.rows, cfg, device)
    report = {"ssl_trunk": probe_scores(rows, "h"), "raw_features": probe_scores(rows, "raw")}
    if args.ref_ckpt:
        ref, rcfg, _ = load_model(args.ref_ckpt, device)
        ref_rows = collect(ref, loader, ds.rows, rcfg, device)
        report["ref_trunk"] = probe_scores(ref_rows, "h")
    print(json.dumps(report, indent=1))
    _write(args, "horizon", report)


# ---------------------------------------------------------------- fog

def cmd_fog(args):
    device = pick_device()
    model, cfg, _ = load_model(args.ckpt, device)
    from .model.masking import sample_and_apply
    ds = GameBundleDataset(args.bundles, "val")
    loader, _ = make_loader(ds, 65536, False, 0, 0)
    gen = torch.Generator().manual_seed(7)
    mask_cfg = MaskConfig(n_champs=1, spans_per_champ_max=1, span_min=60, span_max=60)

    top1 = top5 = persist1 = n = 0
    with torch.no_grad():
        for batch in loader:
            batch = to_device(batch, device)
            # junglers only — the pre-registered test is jungler localization
            masked = sample_and_apply(batch, mask_cfg, gen, champ_slots=JUNGLE_SLOTS)
            cm = masked["champ_mask"]                       # [B,T,10]
            if not cm.any():
                continue
            with autocast_ctx(device, cfg.train.amp_dtype):
                out = model(masked, ssl=True)
            logits = out["recon_pos"]                       # [B,T,10,576]
            true_bin = batch["pos_bin"]                     # [B,T,10]
            for b in range(cm.shape[0]):
                for c in torch.nonzero(cm[b].any(0)).flatten().tolist():
                    ts = torch.nonzero(cm[b, :, c]).flatten()
                    if not len(ts):
                        continue
                    last_seen = true_bin[b, max(int(ts[0]) - 1, 0), c]
                    lg = logits[b, ts, c]
                    tb = true_bin[b, ts, c]
                    top1 += int((lg.argmax(-1) == tb).sum())
                    top5 += int((lg.topk(5, -1).indices == tb[:, None]).any(-1).sum())
                    persist1 += int((tb == last_seen).sum())
                    n += len(ts)
    report = {"n_masked_seconds": n, "top1": top1 / max(n, 1), "top5": top5 / max(n, 1),
              "persistence_top1": persist1 / max(n, 1), "uniform_top1": 1 / 576}
    print(json.dumps(report, indent=1))
    _write(args, "fog", report)


# ---------------------------------------------------------------- event gain

def cmd_event_gain(args):
    with np.load(Path(args.preds) / "test_preds.npz") as z:
        preds = {k: z[k] for k in z.files}
    with np.load(Path(args.ref_preds) / "test_preds.npz") as z:
        ref = {k: z[k] for k in z.files}
    with open(Path(args.bundles) / "bundles_manifest.json", encoding="utf-8") as fh:
        manifest = {g["game"]: g for g in json.load(fh)["games"]}

    window = np.arange(-120, 61)
    curves = {k: [] for k in ("dragon", "baron", "tower")}
    for game, p in preds.items():
        if game not in ref:
            continue
        row = manifest[game]
        y = row["y"]
        with np.load(Path(args.bundles) / row["split"] / f"{game}.npz") as z:
            tt = z["tgt_team"]
        eps = 1e-6
        ll_m = -(y * np.log(np.clip(p, eps, 1)) + (1 - y) * np.log(np.clip(1 - p, eps, 1)))
        r = ref[game]
        ll_r = -(y * np.log(np.clip(r, eps, 1)) + (1 - y) * np.log(np.clip(1 - r, eps, 1)))
        gain = ll_r - ll_m                                  # >0 where model beats ref
        events = {
            "dragon": np.nonzero(np.diff(tt[:, :, 2].sum(1)) > 0.5)[0],
            "baron": np.nonzero(np.diff(tt[:, :, 3].sum(1)) > 0.5)[0],
            "tower": np.nonzero(np.diff(tt[:, :, 1].sum(1)) > 0.5)[0],
        }
        for kind, secs in events.items():
            for s in secs:
                idx = s + window
                ok = (idx >= 0) & (idx < len(gain))
                cur = np.full(len(window), np.nan)
                cur[ok] = gain[idx[ok]]
                curves[kind].append(cur)

    report = {}
    for kind, cs_ in curves.items():
        if not cs_:
            continue
        arr = np.nanmean(np.stack(cs_), 0)
        pre = float(np.nanmean(arr[(window >= -90) & (window <= -30)]))
        report[kind] = {"n_events": len(cs_), "gain_pre_30_90s": pre,
                        "gain_at_take": float(arr[window == 0][0]),
                        "curve": {int(w): float(v) for w, v in zip(window[::15], arr[::15])}}
    print(json.dumps(report, indent=1))
    _write(args, "event_gain", report)


def _write(args, name, report):
    out = Path(getattr(args, "out", None) or "runs/probes")
    out.mkdir(parents=True, exist_ok=True)
    with open(out / f"{name}.json", "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=1)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="task", required=True)
    h = sub.add_parser("horizon")
    h.add_argument("--ckpt", required=True)
    h.add_argument("--ref-ckpt", default=None)
    f = sub.add_parser("fog")
    f.add_argument("--ckpt", required=True)
    e = sub.add_parser("event-gain")
    e.add_argument("--preds", required=True)
    e.add_argument("--ref-preds", required=True)
    for p in (h, f, e):
        p.add_argument("--bundles", default="data/winprob_bundles")
        p.add_argument("--out", default="runs/probes")
    args = ap.parse_args()
    {"horizon": cmd_horizon, "fog": cmd_fog, "event-gain": cmd_event_gain}[args.task](args)


if __name__ == "__main__":
    main()
