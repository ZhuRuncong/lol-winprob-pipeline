"""SSL target construction — always from the CLEAN target tensors on the
original time grid, never from augmented/masked inputs.

Every future-horizon target carries a validity mask requiring t + h < T_g
(true per-game length): targets are masked at game end, never clamped —
clamping would teach an end-of-game detector correlated with the outcome.

Returned dict (suffix _v = validity mask):
  delta_champ [B,T,10,9] + _v      z-normalized future deltas,
                                   feature-major (total_gold,xp,cs) x (5,30,120)
  delta_team  [B,T,6]  + _v        (gold_diff, towers_sum) x horizons
  pos_future  [B,T,10,2] long + _v grid-bin class at t+h for POS_HORIZONS
  hazard_champ [B,T,10] + _v       death within 30s
  hazard_team  [B,T,8]  + _v       (kill15,tower60,dragon60,baron60) x 2 teams,
                                   hazard-major layout
  recon_pos   [B,T,10] long        current grid bin (masked-recon target)
  recon_vals  [B,T,10,5]           hp_frac, mana_frac, level/18,
                                   log1p(total_gold), cs/400
  recon_alive [B,T,10]
  tte         [B,T] + _v           log1p((T_g - t)/60)
  y           [B]
"""
from __future__ import annotations

import torch

from ..config import DELTA_HORIZONS, POS_HORIZONS

# tgt_champ column order (transforms.TGT_CHAMP_FEATURES)
CS, GOLD, XP, HP, MANA, LEVEL, ALIVE = range(7)
# tgt_team column order (transforms.TGT_TEAM_FEATURES)
T_KILLS, T_TOWERS, T_DRAGONS, T_BARON, T_GOLD = range(5)
DELTA_CHAMP_COLS = [GOLD, XP, CS]        # feature-major order in stats/norm_v2
HAZARD_TEAM_SPEC = [(T_KILLS, 15), (T_TOWERS, 60), (T_DRAGONS, 60), (T_BARON, 60)]


def shift_back(x: torch.Tensor, h: int) -> torch.Tensor:
    """x[t+h] with zero right-padding (padded cells are always mask-invalid)."""
    return torch.cat([x[:, h:], torch.zeros_like(x[:, :h])], dim=1)


def build_ssl_targets(batch: dict, norm_stats: dict, device) -> dict:
    tc = batch["tgt_champ"].float()                  # [B,T,10,7]
    tt = batch["tgt_team"].float()                   # [B,T,2,5]
    deaths = batch["tgt_champ_deaths"].float()       # [B,T,10]
    pos_bin = batch["pos_bin"]                       # [B,T,10] long
    valid = batch["valid"]
    B, T = valid.shape
    T_g = valid.sum(1)                               # [B]
    t_idx = torch.arange(T, device=valid.device)[None, :]

    def horizon_valid(h: int) -> torch.Tensor:       # [B,T]
        return valid & ((t_idx + h) < T_g[:, None])

    n_h = len(DELTA_HORIZONS)
    dc_mean = torch.tensor(norm_stats["delta_champ_mean"], dtype=torch.float32,
                           device=device)
    dc_std = torch.tensor(norm_stats["delta_champ_std"], dtype=torch.float32,
                          device=device).clamp_min(1e-3)
    dt_mean = torch.tensor(norm_stats["delta_team_mean"], dtype=torch.float32,
                           device=device)
    dt_std = torch.tensor(norm_stats["delta_team_std"], dtype=torch.float32,
                          device=device).clamp_min(1e-3)

    out: dict[str, torch.Tensor] = {"y": batch["y"]}

    # ---- future deltas -------------------------------------------------
    d_champ = torch.zeros(B, T, 10, len(DELTA_CHAMP_COLS) * n_h, device=device)
    d_champ_v = torch.zeros(B, T, len(DELTA_CHAMP_COLS) * n_h, dtype=torch.bool,
                            device=device)
    gold_diff = tt[:, :, 0, T_GOLD] - tt[:, :, 1, T_GOLD]          # [B,T]
    towers_sum = tt[:, :, :, T_TOWERS].sum(-1)                     # [B,T]
    d_team = torch.zeros(B, T, 2 * n_h, device=device)
    d_team_v = torch.zeros(B, T, 2 * n_h, dtype=torch.bool, device=device)

    for hi, h in enumerate(DELTA_HORIZONS):
        hv = horizon_valid(h)
        for fi, col in enumerate(DELTA_CHAMP_COLS):
            j = fi * n_h + hi
            delta = shift_back(tc[..., col], h) - tc[..., col]     # [B,T,10]
            d_champ[..., j] = (delta - dc_mean[j]) / dc_std[j]
            d_champ_v[..., j] = hv
        for fi, series in enumerate((gold_diff, towers_sum)):
            j = fi * n_h + hi
            delta = shift_back(series, h) - series
            d_team[..., j] = (delta - dt_mean[j]) / dt_std[j]
            d_team_v[..., j] = hv
    out.update(delta_champ=d_champ, delta_champ_v=d_champ_v,
               delta_team=d_team, delta_team_v=d_team_v)

    # ---- future position ----------------------------------------------
    pf = torch.zeros(B, T, 10, len(POS_HORIZONS), dtype=torch.long, device=device)
    pf_v = torch.zeros(B, T, 10, len(POS_HORIZONS), dtype=torch.bool, device=device)
    for hi, h in enumerate(POS_HORIZONS):
        pf[..., hi] = shift_back(pos_bin, h)
        alive_at = shift_back(tc[..., ALIVE], h) > 0.5
        pf_v[..., hi] = horizon_valid(h)[:, :, None] & alive_at
    out.update(pos_future=pf, pos_future_v=pf_v)

    # ---- hazards --------------------------------------------------------
    out["hazard_champ"] = ((shift_back(deaths, 30) - deaths) >= 1.0).float()
    out["hazard_champ_v"] = horizon_valid(30)
    hz = torch.zeros(B, T, 8, device=device)
    hz_v = torch.zeros(B, T, 8, dtype=torch.bool, device=device)
    for k, (col, h) in enumerate(HAZARD_TEAM_SPEC):
        hv = horizon_valid(h)
        for tm in (0, 1):
            series = tt[:, :, tm, col]
            hz[..., 2 * k + tm] = ((shift_back(series, h) - series) >= 1.0).float()
            hz_v[..., 2 * k + tm] = hv
    out.update(hazard_team=hz, hazard_team_v=hz_v)

    # ---- masked-recon current-state targets -----------------------------
    out["recon_pos"] = pos_bin
    out["recon_vals"] = torch.stack([
        tc[..., HP], tc[..., MANA], tc[..., LEVEL] / 18.0,
        torch.log1p(tc[..., GOLD].clamp_min(0)), tc[..., CS] / 400.0,
    ], dim=-1)
    out["recon_alive"] = tc[..., ALIVE]

    # ---- time to end ----------------------------------------------------
    out["tte"] = torch.log1p((T_g[:, None] - t_idx).clamp_min(0).float() / 60.0)
    out["tte_v"] = valid
    return out
