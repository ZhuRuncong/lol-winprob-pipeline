"""Phase A: self-supervised pretraining. Dense SSL objectives fit the trunk;
the win head runs at weight 0.05 as a monitor / mild shaping only (the
label-untouched claim for probes uses checkpoints selected by the frozen
probe, whose own logistic fit never backpropagates into the network).

Checkpoint selection: frozen-probe win log-loss on val (linear probe on h_t),
tiebreak val SSL loss. Auto-resume; one divergence retry at half LR.

Usage:
  python -m pipeline.winprob.train_ssl --config configs/winprob/ssl_M.yaml \
      --bundles data/winprob_bundles --run runs/ssl_M_seed1
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import Config, load_config
from .data.dataset import GameBundleDataset, TokenBudgetBatchSampler, collate
from .model import WinProbModel
from .model.masking import sample_and_apply
from .model.targets import build_ssl_targets
from .model.heads import loss_fn
from .train_common import (EMA, MetricsLog, assert_sdpa_backend, autocast_ctx,
                           cosine_lr, load_norm_version, param_groups,
                           pick_device, save_checkpoint)


def make_loader(ds, budget, shuffle, seed, workers):
    """NO persistent_workers — deliberately: dataset.set_epoch must reach the
    worker processes, and persistent workers keep their epoch-0 pickle forever
    (freezing every per-game augmentation draw for the whole run). Workers
    respawn each epoch and re-pickle the updated dataset."""
    sampler = TokenBudgetBatchSampler(ds.lengths(), budget=budget, shuffle=shuffle, seed=seed)
    return DataLoader(ds, batch_sampler=sampler, collate_fn=collate,
                      num_workers=workers,
                      pin_memory=torch.cuda.is_available()), sampler


def to_device(batch, device):
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


@torch.no_grad()
def val_ssl_loss(model, loader, norm_stats, cfg: Config, device) -> dict[str, float]:
    model.eval()
    gen = torch.Generator().manual_seed(1234)      # fixed masking -> comparable epochs
    totals, n = {}, 0
    for batch in loader:
        batch = to_device(batch, device)
        masked = sample_and_apply(batch, cfg.mask, gen)
        targets = build_ssl_targets(batch, norm_stats, device)
        with autocast_ctx(device, cfg.train.amp_dtype):
            out = model(masked, ssl=True, detach_win=True)
            _, parts = loss_fn(out, targets, cfg.loss, "ssl", batch["valid"],
                               champ_mask=masked.get("champ_mask"))
        for k, v in parts.items():
            totals[k] = totals.get(k, 0.0) + v
        n += 1
    model.train()
    return {f"val_{k}": v / max(n, 1) for k, v in totals.items()}


@torch.no_grad()
def extract_probe_features(model, loader, cfg: Config, device, stride=60):
    """h_t features subsampled per game -> (X, y, game_ids) for the frozen probe."""
    model.eval()
    feats, ys, gids = [], [], []
    for batch in loader:
        batch = to_device(batch, device)
        with autocast_ctx(device, cfg.train.amp_dtype):
            h = model.trunk_features(batch)        # [B,T,d]
        valid = batch["valid"]
        for b in range(h.shape[0]):
            T = int(valid[b].sum())
            idx = torch.arange(0, T, stride)
            feats.append(h[b, idx].float().cpu().numpy())
            ys.append(np.full(len(idx), float(batch["y"][b])))
            gids.append(np.full(len(idx), int(batch["game_idx"][b])))
    model.train()
    return np.concatenate(feats), np.concatenate(ys), np.concatenate(gids)


def frozen_probe_logloss(model, train_loader, val_loader, cfg, device) -> float:
    """Logistic probe on frozen h_t: fit on train games, per-game log-loss on val."""
    from sklearn.linear_model import LogisticRegression
    Xtr, ytr, _ = extract_probe_features(model, train_loader, cfg, device)
    Xva, yva, gva = extract_probe_features(model, val_loader, cfg, device)
    clf = LogisticRegression(C=1.0, max_iter=2000)
    clf.fit(Xtr, ytr)
    p = np.clip(clf.predict_proba(Xva)[:, 1], 1e-6, 1 - 1e-6)
    ll = -(yva * np.log(p) + (1 - yva) * np.log(1 - p))
    per_game = [ll[gva == g].mean() for g in np.unique(gva)]
    return float(np.mean(per_game))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    ap.add_argument("--bundles", default="data/winprob_bundles")
    ap.add_argument("--run", required=True)
    ap.add_argument("--trunk", default=None, help="S|M|L|smoke preset override")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--probe-every", type=int, default=4, help="epochs between frozen probes")
    args = ap.parse_args()

    overrides = {}
    if args.trunk:
        overrides["trunk"] = args.trunk
    if args.seed is not None:
        overrides["train.seed"] = args.seed
    if args.epochs is not None:
        overrides["train.epochs"] = args.epochs
    cfg = load_config(args.config, **overrides)
    cfg.train.phase = "ssl"

    device = pick_device()
    torch.manual_seed(cfg.train.seed)
    np.random.seed(cfg.train.seed)
    assert_sdpa_backend()
    norm_version = load_norm_version(args.bundles)
    with open(Path(args.bundles) / "norm_v2.json", encoding="utf-8") as fh:
        norm_stats = json.load(fh)

    train_ds = GameBundleDataset(args.bundles, "train", cfg.train, seed=cfg.train.seed)
    val_ds = GameBundleDataset(args.bundles, "val")
    train_loader, train_sampler = make_loader(train_ds, cfg.train.token_budget, True,
                                              cfg.train.seed, cfg.train.num_workers)
    # eval datasets carry no augmentation/epoch state -> workers are safe here
    eval_workers = min(cfg.train.num_workers, 4)
    val_loader, _ = make_loader(val_ds, cfg.train.token_budget, False, 0, eval_workers)
    # probe fit uses un-augmented train features on a capped subset
    probe_ds = GameBundleDataset(args.bundles, "train")
    probe_ds.rows = probe_ds.rows[:200]
    probe_loader, _ = make_loader(probe_ds, cfg.train.token_budget, False, 0, eval_workers)

    model = WinProbModel(cfg.model).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model params: {n_params/1e6:.2f}M  device: {device}  norm: {norm_version}")

    opt = torch.optim.AdamW(param_groups(model, cfg.train.weight_decay),
                            lr=cfg.train.lr, betas=cfg.train.betas, eps=1e-8)
    ema = EMA(model, cfg.train.ema_decay)
    run_dir = Path(args.run)
    log = MetricsLog(run_dir)
    scaler = torch.amp.GradScaler("cuda", enabled=(cfg.train.amp_dtype == "fp16"
                                                   and device.type == "cuda"))

    steps_per_epoch = len(train_sampler)
    total_steps = steps_per_epoch * cfg.train.epochs
    warmup = int(total_steps * cfg.train.warmup_frac)
    start_epoch, step, best, lr_scale, retried = 0, 0, float("inf"), 1.0, False

    last_ckpt = run_dir / "last.pt"
    if args.resume and last_ckpt.exists():
        ck = torch.load(last_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        if ck.get("opt"):
            opt.load_state_dict(ck["opt"])
        ema.shadow = {k: v.to(device) for k, v in ck["ema"].items()}
        start_epoch = ck["epoch"] + 1
        step = ck.get("step", start_epoch * steps_per_epoch)
        best = ck.get("best", best)
        lr_scale = ck.get("lr_scale", 1.0)
        print(f"resumed from epoch {ck['epoch']} (step {step})")
    else:
        # epoch -1 checkpoint so the divergence-retry path always has a
        # restore point, even if the very first epoch diverges
        save_checkpoint(last_ckpt, model, ema, opt, -1, cfg, norm_version,
                        {"best": best, "lr_scale": lr_scale, "step": 0})

    mask_gen = torch.Generator().manual_seed(cfg.train.seed)
    best_val_at_best = float("inf")
    probes_since_best = 0
    epoch = start_epoch
    while epoch < cfg.train.epochs:
        train_ds.set_epoch(epoch)
        train_sampler.set_epoch(epoch)
        nominal_games = len(train_ds) / max(len(train_sampler), 1)
        epoch_parts, n_batches, diverged = {}, 0, False

        for batch in train_loader:
            lr = cosine_lr(step, total_steps, warmup, cfg.train.lr) * lr_scale
            for g in opt.param_groups:
                g["lr"] = lr
            batch = to_device(batch, device)
            masked = sample_and_apply(batch, cfg.mask, mask_gen)
            targets = build_ssl_targets(batch, norm_stats, device)
            with autocast_ctx(device, cfg.train.amp_dtype):
                out = model(masked, ssl=True, detach_win=True)
                total, parts = loss_fn(out, targets, cfg.loss, "ssl", batch["valid"],
                                       champ_mask=masked.get("champ_mask"),
                                       nominal_games=nominal_games)
            if not torch.isfinite(total):
                diverged = True
                break
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
                epoch_parts[k] = epoch_parts.get(k, 0.0) + v

        if diverged:
            if retried:
                raise SystemExit("diverged twice — killing this arm (per Day-1 gate rule)")
            retried, lr_scale = True, lr_scale * 0.5
            ck = torch.load(last_ckpt, map_location=device, weights_only=False)
            model.load_state_dict(ck["model"])
            if ck.get("opt"):
                opt.load_state_dict(ck["opt"])
            ema.shadow = {k: v.to(device) for k, v in ck["ema"].items()}
            step = ck.get("step", 0)
            epoch = ck["epoch"] + 1        # retry the diverged stretch, not skip it
            print(f"DIVERGED: reloaded epoch-{ck['epoch']} checkpoint, "
                  f"lr_scale=0.5, retrying epoch {epoch}")
            continue

        row = {k: v / max(n_batches, 1) for k, v in epoch_parts.items()}
        backup = ema.copy_to(model)
        row.update(val_ssl_loss(model, val_loader, norm_stats, cfg, device))
        # checkpoint selection happens ONLY at probe epochs: the probe metric
        # (val win log-loss of a frozen linear head) and val SSL loss live on
        # different scales, so they must never be compared to each other.
        selection = None
        if epoch % args.probe_every == args.probe_every - 1 or epoch == cfg.train.epochs - 1:
            row["probe_logloss"] = frozen_probe_logloss(model, probe_loader, val_loader,
                                                        cfg, device)
            selection = row["probe_logloss"]
        EMA.restore(model, backup)

        log.log(epoch=epoch, lr=lr, **row)
        save_checkpoint(last_ckpt, model, ema, opt, epoch, cfg, norm_version,
                        {"best": best, "lr_scale": lr_scale, "step": step})
        if selection is not None:
            val_now = row.get("val_total", float("inf"))
            improved = (selection < best - 1e-6
                        or (abs(selection - best) <= 1e-6 and val_now < best_val_at_best))
            if improved:
                best, best_val_at_best, probes_since_best = selection, val_now, 0
                save_checkpoint(run_dir / "best.pt", model, ema, None, epoch, cfg,
                                norm_version, {"selection": selection})
                print(f"  new best ({selection:.5f}) at epoch {epoch}")
            else:
                probes_since_best += 1
                if probes_since_best >= 3:     # plan: probe-metric patience 3
                    print(f"early stop: no probe improvement in 3 evaluations "
                          f"(epoch {epoch})")
                    break
        epoch += 1

    print(f"done. best selection metric: {best:.5f}  run dir: {run_dir}")


if __name__ == "__main__":
    main()
