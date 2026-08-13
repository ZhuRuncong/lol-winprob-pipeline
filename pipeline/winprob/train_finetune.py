"""Phase B: win-head finetune from a phase-A checkpoint -> one ensemble member.

Clean inputs (no fog masking), plain unsmoothed per-game-averaged BCE with
stride subsampling, retained future-prediction SSL at 0.25x, two LR groups
(backbone low, heads high). Early stopping on val per-game log-loss in the
10-25 minute window. Emits member checkpoint + standard prediction files.

Usage:
  python -m pipeline.winprob.train_finetune --init runs/ssl_M/best.pt \
      --bundles data/winprob_bundles --run runs/ft_s7 --seed 7 \
      --preds-out results/member_s7
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .config import load_config
from .data.dataset import GameBundleDataset
from .model.targets import build_ssl_targets
from .model.heads import loss_fn
from .predict import load_model, predict_split, write_preds, SwappedView
from .train_common import (EMA, MetricsLog, autocast_ctx, cosine_lr,
                           load_norm_version, pick_device, save_checkpoint)
from .train_ssl import make_loader, to_device

WINDOW_1025 = (600, 1500)   # the decision window: 10-25 min


@torch.no_grad()
def val_window_logloss(model, loader, cfg, device, window=WINDOW_1025) -> float:
    """Per-game log-loss restricted to the 10-25 min window, macro over games."""
    model.eval()
    per_game = []
    for batch in loader:
        batch = to_device(batch, device)
        with autocast_ctx(device, cfg.train.amp_dtype):
            out = model(batch, ssl=False)
        logit = out["win_logit"].float()
        bce = torch.nn.functional.binary_cross_entropy_with_logits(
            logit, batch["y"][:, None].expand_as(logit), reduction="none")
        T = logit.shape[1]
        tmask = torch.zeros(T, dtype=torch.bool, device=device)
        tmask[window[0]:min(window[1], T)] = True
        m = batch["valid"] & tmask[None, :]
        for b in range(logit.shape[0]):
            if m[b].any():
                per_game.append(float(bce[b][m[b]].mean()))
    model.train()
    return float(np.mean(per_game)) if per_game else float("inf")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--init", default=None,
                    help="phase-A checkpoint (best.pt); omit for the no-SSL "
                         "from-scratch control (requires --config/--trunk)")
    ap.add_argument("--trunk", default=None, help="S|M|L|smoke (from-scratch mode)")
    ap.add_argument("--config", default=None)
    ap.add_argument("--bundles", default="data/winprob_bundles")
    ap.add_argument("--run", required=True)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--label-fraction", type=float, default=1.0,
                    help="seeded fraction of train SERIES to keep (ablation arm)")
    ap.add_argument("--resume", action="store_true",
                    help="continue from <run>/last.pt if present (spot-safe)")
    ap.add_argument("--preds-out", default=None,
                    help="write val/test predictions here when done")
    args = ap.parse_args()

    device = pick_device()
    # seed BEFORE any model construction so the from-scratch (no --init)
    # control's random initialization is reproducible per --seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.init:
        model, cfg, ckpt_norm = load_model(args.init, device, use_ema=True)
    else:                                   # no-SSL from-scratch control
        from .model import WinProbModel
        overrides = {"trunk": args.trunk} if args.trunk else {}
        cfg = load_config(args.config, **overrides)
        model = WinProbModel(cfg.model).to(device)
        ckpt_norm = "?"
        print("from-scratch mode: random-init model (the no-SSL ablation arm)")
    if args.config:
        file_cfg = load_config(args.config)
        cfg.train = file_cfg.train          # finetune recipe from config file
        cfg.loss = file_cfg.loss
    cfg.train.phase = "finetune"
    cfg.train.seed = args.seed
    if args.epochs is not None:
        cfg.train.epochs = args.epochs

    norm_version = load_norm_version(args.bundles)
    if ckpt_norm not in ("?", norm_version):
        raise SystemExit(f"norm mismatch: init {ckpt_norm} vs bundles {norm_version}")
    with open(Path(args.bundles) / "norm_v2.json", encoding="utf-8") as fh:
        norm_stats = json.load(fh)

    train_ds = GameBundleDataset(args.bundles, "train", cfg.train, seed=cfg.train.seed)
    if args.label_fraction < 1.0:
        from .data.splits import subsample_series
        train_ds.rows = subsample_series(train_ds.rows, args.label_fraction,
                                         seed=args.seed)
        print(f"label-fraction {args.label_fraction}: {len(train_ds.rows)} train games")
    val_ds = GameBundleDataset(args.bundles, "val")
    train_loader, train_sampler = make_loader(train_ds, cfg.train.token_budget, True,
                                              cfg.train.seed, cfg.train.num_workers)
    val_loader, _ = make_loader(val_ds, cfg.train.token_budget, False, 0,
                                min(cfg.train.num_workers, 4))

    backbone, heads = [], []
    for name, p in model.named_parameters():
        (backbone if name.startswith(("encoder.", "trunk.")) else heads).append(p)
    opt = torch.optim.AdamW(
        [{"params": backbone, "lr": cfg.train.backbone_lr, "weight_decay": cfg.train.weight_decay},
         {"params": heads, "lr": cfg.train.lr, "weight_decay": cfg.train.weight_decay}],
        betas=cfg.train.betas, eps=1e-8)
    ema = EMA(model, cfg.train.ema_decay)
    run_dir = Path(args.run)
    log = MetricsLog(run_dir)
    scaler = torch.amp.GradScaler("cuda", enabled=(cfg.train.amp_dtype == "fp16"
                                                   and device.type == "cuda"))
    model.train()

    steps_per_epoch = len(train_sampler)
    total_steps = steps_per_epoch * cfg.train.epochs
    warmup = int(total_steps * cfg.train.warmup_frac)
    base_lrs = [cfg.train.backbone_lr, cfg.train.lr]
    step, best, bad_epochs, start_epoch = 0, float("inf"), 0, 0

    last_ckpt = run_dir / "last.pt"
    if args.resume and last_ckpt.exists():
        ck = torch.load(last_ckpt, map_location=device, weights_only=False)
        if ck.get("norm_version") not in (None, norm_version):
            raise SystemExit(
                f"checkpoint {last_ckpt} was trained on norm "
                f"{ck.get('norm_version')} but bundles are {norm_version} — "
                f"feature contract changed; delete the old run dir to start fresh")
        model.load_state_dict(ck["model"])
        if ck.get("opt"):
            opt.load_state_dict(ck["opt"])
        ema.shadow = {k: v.to(device) for k, v in ck["ema"].items()}
        start_epoch = ck["epoch"] + 1
        step = ck.get("step", start_epoch * steps_per_epoch)
        best = ck.get("best", best)
        bad_epochs = ck.get("bad_epochs", 0)
        print(f"resumed member from epoch {ck['epoch']} (best {best:.5f})")

    for epoch in range(start_epoch, cfg.train.epochs):
        train_ds.set_epoch(epoch)
        train_sampler.set_epoch(epoch)
        nominal_games = len(train_ds) / max(len(train_sampler), 1)
        offset = epoch % max(cfg.train.win_stride, 1)   # fresh stride offset per epoch
        parts_sum, n_batches = {}, 0

        for batch in train_loader:
            for g, base in zip(opt.param_groups, base_lrs):
                g["lr"] = cosine_lr(step, total_steps, warmup, base)
            batch = to_device(batch, device)
            targets = build_ssl_targets(batch, norm_stats, device)
            with autocast_ctx(device, cfg.train.amp_dtype):
                out = model(batch, ssl=True)            # clean inputs, SSL heads retained
                total, parts = loss_fn(out, targets, cfg.loss, "finetune", batch["valid"],
                                       win_stride=cfg.train.win_stride,
                                       stride_offset=offset,
                                       quarter_balance=cfg.train.quarter_balance,
                                       nominal_games=nominal_games)
            if not torch.isfinite(total):
                raise SystemExit(f"non-finite finetune loss at epoch {epoch}")
            opt.zero_grad(set_to_none=True)
            scaler.scale(total).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
            scaler.step(opt)
            scaler.update()
            ema.update(model)
            step += 1
            n_batches += 1
            for k, v in parts.items():
                parts_sum[k] = parts_sum.get(k, 0.0) + v

        backup = ema.copy_to(model)
        val_ll = val_window_logloss(model, val_loader, cfg, device)
        EMA.restore(model, backup)
        log.log(epoch=epoch, val_1025_logloss=val_ll,
                **{k: v / max(n_batches, 1) for k, v in parts_sum.items()})

        if val_ll < best:
            best, bad_epochs = val_ll, 0
            save_checkpoint(run_dir / "best.pt", model, ema, None, epoch, cfg,
                            norm_version, {"val_1025_logloss": val_ll, "init": args.init,
                                           "label_fraction": args.label_fraction})
        else:
            bad_epochs += 1
        # written every epoch so a reclaim loses at most one epoch
        save_checkpoint(last_ckpt, model, ema, opt, epoch, cfg, norm_version,
                        {"best": best, "step": step, "bad_epochs": bad_epochs})
        if bad_epochs >= cfg.train.patience:
            print(f"early stop at epoch {epoch} (patience {cfg.train.patience})")
            break

    print(f"member done: best val 10-25min log-loss {best:.5f}")

    if args.preds_out:
        member, mcfg, _ = load_model(run_dir / "best.pt", device, use_ema=True)
        out = Path(args.preds_out)
        for split in ("val", "test"):
            ds = GameBundleDataset(args.bundles, split)
            write_preds(out, split, predict_split(member, ds, mcfg, device))
            write_preds(out, split, predict_split(member, SwappedView(ds), mcfg, device),
                        swap=True)
        with open(out / "meta.json", "w", encoding="utf-8") as fh:
            json.dump({"model": str(run_dir / "best.pt"), "init": args.init,
                       "seed": cfg.train.seed, "val_1025_logloss": best,
                       "label_fraction": args.label_fraction,
                       "norm_version": norm_version}, fh, indent=1)
        print(f"predictions written to {out}")


if __name__ == "__main__":
    main()
