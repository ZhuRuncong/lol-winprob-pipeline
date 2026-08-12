"""Bundle the shipped predictor into ONE model.pt file.

Contents: EMA weights of every ensemble member, the model config, the
train-split normalization stats, and the fitted calibration parameters —
everything inference needs. Load side:

    ck = torch.load("model.pt", map_location="cpu", weights_only=False)
    models = []
    for sd in ck["members"]:
        m = WinProbModel(ModelConfig(**ck["model_cfg"])); m.load_state_dict(sd)
        models.append(m.eval())
    # p = mean over members of sigmoid(win_logit); apply ck["calibration"]
    # ("selected" + temperatures) to the mean and swap streams, then
    # symmetrize: p* = (c(p) + 1 - c(p_swap)) / 2.

Usage:
  python -m pipeline.winprob.export_model --members runs/ft_s1/best.pt ... \
      --bundles data/winprob_bundles \
      --calibration results/ensemble_cal/calibration_report.json --out model.pt
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import torch

from .predict import load_model
from .train_common import git_sha


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--members", nargs="+", required=True,
                    help="phase-B member checkpoints (best.pt)")
    ap.add_argument("--bundles", default="data/winprob_bundles")
    ap.add_argument("--calibration", default=None,
                    help="calibration_report.json from calibrate.py")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    device = torch.device("cpu")
    members, model_cfg, norm_version = [], None, None
    for path in args.members:
        model, cfg, nv = load_model(path, device, use_ema=True)
        members.append({k: v.cpu() for k, v in model.state_dict().items()})
        from dataclasses import asdict
        model_cfg = model_cfg or asdict(cfg.model)
        norm_version = norm_version or nv

    with open(Path(args.bundles) / "norm_v2.json", encoding="utf-8") as fh:
        norm_stats = json.load(fh)
    if norm_version not in (None, "?", norm_stats["norm_version"]):
        raise SystemExit(f"norm mismatch: members {norm_version} vs "
                         f"bundles {norm_stats['norm_version']}")

    calibration = None
    if args.calibration:
        with open(args.calibration, encoding="utf-8") as fh:
            calibration = json.load(fh)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "members": members,
        "model_cfg": model_cfg,
        "norm_stats": norm_stats,
        "calibration": calibration,
        "member_checkpoints": args.members,
        "git_sha": git_sha(),
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }, out)
    size_mb = out.stat().st_size / 1e6
    print(f"wrote {out}  ({len(members)} members, {size_mb:.1f} MB, "
          f"norm {norm_stats['norm_version']})")


if __name__ == "__main__":
    main()
