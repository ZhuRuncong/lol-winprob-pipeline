"""The leakage contract as executable tests.

Run: python -m pytest tests/test_winprob_contracts.py -x -q
Requires the packed local sample (data/winprob_bundles_local) — build it with:
  python -m pipeline.winprob.data.stats --local-sample --out data/winprob_bundles_local/norm_v2.json
  python -m pipeline.winprob.data.pack  --local-sample --out data/winprob_bundles_local
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from pipeline.winprob.config import MaskConfig, TrainConfig, load_config
from pipeline.winprob.data.dataset import (GameBundleDataset, SWAP_PERM,
                                           TokenBudgetBatchSampler, collate)
from pipeline.winprob.model import WinProbModel, loss_fn
from pipeline.winprob.model.masking import sample_and_apply
from pipeline.winprob.model.targets import build_ssl_targets

ROOT = Path(__file__).resolve().parent.parent
BUNDLES = ROOT / "data" / "winprob_bundles_local"

pytestmark = pytest.mark.skipif(
    not (BUNDLES / "bundles_manifest.json").exists(),
    reason="local bundles not packed (see module docstring)")


@pytest.fixture(scope="module")
def batch():
    ds = GameBundleDataset(BUNDLES, "val")
    sampler = TokenBudgetBatchSampler(ds.lengths(), budget=4096, shuffle=False)
    return next(iter(DataLoader(ds, batch_sampler=sampler, collate_fn=collate)))


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    return WinProbModel(load_config(None, trunk="smoke").model).eval()


@pytest.fixture(scope="module")
def norm_stats():
    with open(BUNDLES / "norm_v2.json", encoding="utf-8") as fh:
        return json.load(fh)


# ---------------------------------------------------------------- causality

def _corrupt_after(batch, t0):
    torch.manual_seed(1)
    c = dict(batch)
    for k in ("champ", "team", "glob", "item_cd"):
        x = batch[k].clone().float()
        x[:, t0:] = torch.randn_like(x[:, t0:])
        c[k] = x
    it = batch["items"].clone()
    it[:, t0:] = torch.randint(0, 4095, it[:, t0:].shape)
    c["items"] = it
    return c


def test_causality_full_attention(batch, model):
    t0 = 40
    with torch.no_grad():
        a = model(batch, ssl=False)["win_logit"]
        b = model(_corrupt_after(batch, t0), ssl=False)["win_logit"]
    assert torch.allclose(a[:, :t0], b[:, :t0], atol=1e-5)


def test_causality_windowed_attention(batch):
    cfg = load_config(None, trunk="smoke")
    cfg.model.attn_window = 16
    torch.manual_seed(0)
    m = WinProbModel(cfg.model).eval()
    t0 = 40
    with torch.no_grad():
        a = m(batch, ssl=False)["win_logit"]
        b = m(_corrupt_after(batch, t0), ssl=False)["win_logit"]
    assert torch.allclose(a[:, :t0], b[:, :t0], atol=1e-5)


# ---------------------------------------------------------------- masking

def test_masking_zeroes_all_input_paths_and_preserves_originals(batch):
    gen = torch.Generator().manual_seed(5)
    orig = {k: v.clone() for k, v in batch.items()}
    masked = sample_and_apply(batch, MaskConfig(), gen)
    cm = masked["champ_mask"]
    assert cm.any()
    assert (masked["champ"][cm] == 0).all()
    assert (masked["items"][cm] == 0).all()
    assert (masked["item_cd"][cm] == 0).all()
    for k, v in orig.items():                      # caller's batch untouched
        assert torch.equal(batch[k], v), k
    for k in ("tgt_champ", "tgt_team", "tgt_champ_deaths", "pos_bin"):
        assert torch.equal(masked[k], batch[k]), f"{k} must never be masked"


# ---------------------------------------------------------------- swap

def test_swap_roundtrip_and_label_flip():
    ds = GameBundleDataset(BUNDLES, "val")
    d0 = ds[0]
    d1 = copy.deepcopy(d0)
    GameBundleDataset._swap(d1)
    assert int(d1["y"]) == 1 - int(d0["y"])
    assert np.array_equal(d1["champ"][:, :5], d0["champ"][:, 5:])
    assert np.array_equal(d1["team"][:, 0], d0["team"][:, 1])
    assert (np.asarray(d1["side"]) == (np.asarray(d0["side"])[SWAP_PERM] ^ 1)).all()
    GameBundleDataset._swap(d1)
    for k in d0:
        assert np.array_equal(np.asarray(d0[k]), np.asarray(d1[k])), k


# ---------------------------------------------------------------- ward t_end ban

def test_ward_raster_uses_no_future_information():
    """Truncating the ward table's future must not change the raster's past.

    If any feature value at time t derived from t_end (e.g. remaining
    lifetime), cutting intervals short at t would change values before t.
    """
    from pipeline.winprob.data.transforms import ward_raster
    rng = np.random.default_rng(0)
    W, T = 60, 400
    wards = np.stack([
        rng.integers(0, 2, W), rng.integers(0, 4, W),
        rng.random(W), rng.random(W),
        rng.uniform(0, 300, W), np.zeros(W)], axis=1)
    wards[:, 5] = wards[:, 4] + rng.uniform(5, 200, W)     # t_end > t_start
    cut = 200
    full = ward_raster(wards, T)
    truncated = wards.copy()
    truncated[:, 5] = np.minimum(truncated[:, 5], cut)     # kill everything at t=cut
    trunc = ward_raster(truncated, T)
    assert np.array_equal(full[:cut], trunc[:cut])


# ---------------------------------------------------------------- horizons

def test_ssl_targets_masked_never_clamped_at_game_end(batch, norm_stats):
    targets = build_ssl_targets(batch, norm_stats, torch.device("cpu"))
    T_g = batch["valid"].sum(1)
    for b in range(batch["valid"].shape[0]):
        tg = int(T_g[b])
        # the last h seconds of each game must be invalid for horizon h
        assert not targets["delta_champ_v"][b, tg - 3:, :].any()      # h>=5 all invalid
        assert not targets["hazard_champ_v"][b, tg - 20:tg, ].any()   # h=30
        assert targets["delta_champ_v"][b, : max(tg - 121, 0)].all()  # all valid before


# ---------------------------------------------------------------- norm provenance

def test_norm_v2_provenance():
    from pipeline.winprob.data.stats import load_stats
    s = load_stats(BUNDLES / "norm_v2.json")
    assert s["norm_version"].startswith("v2-")
    with open(BUNDLES / "bundles_manifest.json", encoding="utf-8") as fh:
        bm = json.load(fh)
    assert bm["norm_version"] == s["norm_version"]
    with pytest.raises(SystemExit):                 # v1 corpus-wide stats refused
        load_stats(ROOT / "data" / "pack" / "norm.json")


def test_no_game_length_leak_in_inputs(batch):
    """No input feature may encode eventual game length: two games of
    different length must have identical global features at the same t."""
    glob = batch["glob"].float()
    T_g = batch["valid"].sum(1)
    if len(T_g) < 2 or int(T_g.min()) == int(T_g.max()):
        pytest.skip("need two games of different length in the batch")
    # clock features (cols 6:12) are pure functions of t
    a, b = 0, 1
    t = min(int(T_g[a]), int(T_g[b])) - 1
    assert torch.allclose(glob[a, t, 6:], glob[b, t, 6:], atol=1e-3)


# ---------------------------------------------------------------- label-free phase A

def test_phase_a_label_never_shapes_trunk(batch, norm_stats):
    """The phase-A win term is a monitor only: with every SSL weight zeroed,
    backprop through the ssl-phase total must leave encoder and trunk with
    zero gradient. This is the label-free-checkpoint property the hypothesis
    probes depend on — regression test for a confirmed review finding."""
    from pipeline.winprob.config import LossWeights
    m = WinProbModel(load_config(None, trunk="smoke").model)
    gen = torch.Generator().manual_seed(3)
    masked = sample_and_apply(batch, MaskConfig(), gen)
    targets = build_ssl_targets(batch, norm_stats, torch.device("cpu"))
    w = LossWeights(mask=0.0, delta=0.0, pos=0.0, hazard=0.0, tte=0.0)
    out = m(masked, ssl=True, detach_win=True)
    total, _ = loss_fn(out, targets, w, "ssl", batch["valid"],
                       champ_mask=masked["champ_mask"])
    total.backward()
    for n, p in m.named_parameters():
        if n.startswith(("encoder.", "trunk.")):
            assert p.grad is None or torch.all(p.grad == 0), f"label grad reached {n}"
    assert any(p.grad is not None and p.grad.abs().sum() > 0
               for n, p in m.named_parameters() if n.startswith("win_head."))


# ---------------------------------------------------------------- loss sanity

def test_losses_finite_and_backprop(batch, model, norm_stats):
    m = WinProbModel(load_config(None, trunk="smoke").model)
    gen = torch.Generator().manual_seed(2)
    masked = sample_and_apply(batch, MaskConfig(), gen)
    targets = build_ssl_targets(batch, norm_stats, torch.device("cpu"))
    cfg = load_config(None, trunk="smoke")
    out = m(masked, ssl=True)
    total, _ = loss_fn(out, targets, cfg.loss, "ssl", batch["valid"],
                       champ_mask=masked["champ_mask"])
    assert torch.isfinite(total)
    total.backward()
    assert all(p.grad is not None for p in m.parameters() if p.requires_grad)
