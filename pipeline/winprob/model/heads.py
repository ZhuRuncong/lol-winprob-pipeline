"""WinProbModel assembly + the phase-aware loss.

The win head reads trunk output h_t ONLY — there is no encoder-to-win-head
bypass, so label gradient reaches the encoder only through the (low-LR
finetuned) trunk. Per-champion SSL heads read concat[h_t, C[t,i]] through one
shared projection.

Loss accounting: the win BCE is averaged WITHIN each game first (optionally
quarter-balanced, optionally stride-subsampled) and then across games — a
game is one unit of supervision regardless of length. No label smoothing
anywhere: the deliverable is a calibrated probability.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from ..config import (DELTA_HORIZONS, LossWeights, ModelConfig, POS_BINS,
                      POS_HORIZONS)
from .encoder import StateEncoder
from .temporal import CausalTrunk

N_DELTA_CHAMP = 3 * len(DELTA_HORIZONS)      # 9
N_DELTA_TEAM = 2 * len(DELTA_HORIZONS)       # 6
N_POS_H = len(POS_HORIZONS)                  # 2


class WinProbModel(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.encoder = StateEncoder(cfg)
        self.trunk = CausalTrunk(cfg)
        d = cfg.d_model
        self.win_head = nn.Sequential(
            nn.LayerNorm(d), nn.Linear(d, cfg.win_head_hidden), nn.GELU(),
            nn.Dropout(cfg.win_head_dropout), nn.Linear(cfg.win_head_hidden, 1))
        # shared per-champion SSL projection over concat[h_t, C[t,i]]
        self.champ_proj = nn.Sequential(nn.Linear(d + cfg.d_tok, 128), nn.GELU())
        self.head_recon_pos = nn.Linear(128, POS_BINS)
        self.head_recon_vals = nn.Linear(128, 5)
        self.head_recon_alive = nn.Linear(128, 1)
        self.head_delta_champ = nn.Linear(128, N_DELTA_CHAMP)
        self.head_pos_future = nn.Linear(128, N_POS_H * POS_BINS)
        self.head_hazard_champ = nn.Linear(128, 1)
        self.head_delta_team = nn.Linear(d, N_DELTA_TEAM)
        self.head_hazard_team = nn.Linear(d, 8)
        self.head_tte = nn.Linear(d, 1)

    def trunk_features(self, batch: dict) -> torch.Tensor:
        """h_t only (encoder + trunk, no heads); batch used as given."""
        s, _ = self.encoder(batch)
        return self.trunk(s, batch["valid"])

    def forward(self, batch: dict, ssl: bool = False,
                detach_win: bool = False) -> dict:
        """detach_win=True (phase A only): the win head reads h.detach(), so
        the label-monitor loss can NEVER shape the trunk or encoder — the
        phase-A checkpoint stays label-free, which the hypothesis probes
        require. Phase B passes detach_win=False so the label finetunes the
        trunk at its low LR."""
        s, C = self.encoder(batch)
        h = self.trunk(s, batch["valid"])            # [B,T,d]
        out = {"win_logit": self.win_head(h.detach() if detach_win else h).squeeze(-1)}
        if not ssl:
            return out
        B, T, _, dt = C.shape
        cf = self.champ_proj(torch.cat(
            [h[:, :, None].expand(B, T, 10, h.shape[-1]), C], dim=-1))  # [B,T,10,128]
        out["recon_pos"] = self.head_recon_pos(cf)
        out["recon_vals"] = self.head_recon_vals(cf)
        out["recon_alive"] = self.head_recon_alive(cf).squeeze(-1)
        out["delta_champ"] = self.head_delta_champ(cf)
        out["pos_future"] = self.head_pos_future(cf).view(B, T, 10, N_POS_H, POS_BINS)
        out["hazard_champ"] = self.head_hazard_champ(cf).squeeze(-1)
        out["delta_team"] = self.head_delta_team(h)
        out["hazard_team"] = self.head_hazard_team(h)
        out["tte"] = self.head_tte(h).squeeze(-1)
        return out


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------

def _masked_huber(pred, target, mask):
    if not mask.any():
        return pred.new_zeros(())
    return F.huber_loss(pred[mask].float(), target[mask].float(), delta=1.0)


def _masked_bce(logit, target, mask):
    if not mask.any():
        return logit.new_zeros(())
    return F.binary_cross_entropy_with_logits(logit[mask].float(), target[mask].float())


def _masked_ce(logits, target, mask):
    if not mask.any():
        return logits.new_zeros(())
    return F.cross_entropy(logits[mask].float(), target[mask])


def win_loss(win_logit, y, valid, stride=1, offset=0, quarter_balance=False,
             nominal_games=None):
    """Per-game-averaged BCE with stride subsampling and quarter balancing.

    nominal_games: divide the per-game loss SUM by this constant instead of
    the batch's own game count. Token-budget batches hold few long games or
    many short ones; without a fixed denominator a game's gradient weight
    would scale with its batchmates' lengths.
    """
    B, T = win_logit.shape
    bce = F.binary_cross_entropy_with_logits(
        win_logit.float(), y[:, None].expand(B, T).float(), reduction="none")
    t_idx = torch.arange(T, device=win_logit.device)[None, :]
    sel = valid & (((t_idx + offset) % max(stride, 1)) == 0)

    per_game = []
    T_g = valid.sum(1)
    for b in range(B):
        m = sel[b]
        if not m.any():
            m = valid[b]                              # degenerate: use all seconds
        if quarter_balance:
            q = (t_idx[0] * 4 // T_g[b].clamp_min(1)).clamp(max=3)
            vals = []
            for qi in range(4):
                mq = m & (q == qi)
                if mq.any():
                    vals.append(bce[b][mq].mean())
            per_game.append(torch.stack(vals).mean())
        else:
            per_game.append(bce[b][m].mean())
    return torch.stack(per_game).sum() / (nominal_games or B)


def loss_fn(outputs: dict, targets: dict, w: LossWeights, phase: str,
            valid: torch.Tensor, champ_mask: torch.Tensor | None = None,
            win_stride: int = 1, stride_offset: int = 0,
            quarter_balance: bool = False, nominal_games: float | None = None):
    parts: dict[str, float] = {}
    v3 = valid[:, :, None]                            # broadcast over champs

    l_win = win_loss(outputs["win_logit"], targets["y"], valid,
                     stride=win_stride, offset=stride_offset,
                     quarter_balance=quarter_balance, nominal_games=nominal_games)
    parts["win"] = float(l_win.detach())

    # delta_champ_v is [B,T,9] (validity is per-horizon, shared across champs)
    l_delta = (_masked_huber(outputs["delta_champ"], targets["delta_champ"],
                             targets["delta_champ_v"][:, :, None, :].expand_as(
                                 outputs["delta_champ"]))
               + _masked_huber(outputs["delta_team"], targets["delta_team"],
                               targets["delta_team_v"]))
    parts["delta"] = float(l_delta.detach())

    l_pos = _masked_ce(outputs["pos_future"], targets["pos_future"],
                       targets["pos_future_v"])
    parts["pos"] = float(l_pos.detach())

    l_hazard = (_masked_bce(outputs["hazard_champ"], targets["hazard_champ"],
                            targets["hazard_champ_v"][:, :, None].expand_as(
                                outputs["hazard_champ"]))
                + _masked_bce(outputs["hazard_team"], targets["hazard_team"],
                              targets["hazard_team_v"]))
    parts["hazard"] = float(l_hazard.detach())

    if phase == "ssl":
        if champ_mask is None:
            raise ValueError("phase 'ssl' requires champ_mask")
        cells = champ_mask & v3
        l_mask = (_masked_ce(outputs["recon_pos"], targets["recon_pos"], cells)
                  + _masked_huber(outputs["recon_vals"], targets["recon_vals"],
                                  cells[..., None].expand_as(outputs["recon_vals"]))
                  + _masked_bce(outputs["recon_alive"], targets["recon_alive"], cells))
        l_tte = _masked_huber(outputs["tte"], targets["tte"], targets["tte_v"])
        parts["mask"] = float(l_mask.detach())
        parts["tte"] = float(l_tte.detach())
        total = (w.mask * l_mask + w.delta * l_delta + w.pos * l_pos
                 + w.hazard * l_hazard + w.tte * l_tte + 0.05 * l_win)
    elif phase == "finetune":
        total = (w.win * l_win
                 + w.ssl_retention * (l_delta + 0.5 * l_pos + 0.5 * l_hazard))
    else:
        raise ValueError(f"unknown phase {phase!r}")
    parts["total"] = float(total.detach())
    return total, parts
