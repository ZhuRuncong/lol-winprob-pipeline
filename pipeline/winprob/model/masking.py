"""Fog-of-war masking: hide champions' dynamic inputs over sampled spans.

Contract (unit-tested): the returned batch is a shallow copy whose champ /
items / item_cd tensors are CLONES with masked cells zeroed — the caller's
batch is never mutated, and because the encoder's token path AND skip path
both read these same tensors, no trace of a masked champion's true features
can reach any input path. Statics stay visible (you always know WHO is
missing). Target tensors are never touched.
"""
from __future__ import annotations

import torch

from ..config import MaskConfig


def apply_spans(batch: dict, spans: list[tuple[int, int, int, int]]) -> dict:
    """Apply explicit fog spans [(b, champ, t0, t1), ...] — the single zeroing
    implementation every masking consumer (training, fog probe) shares."""
    champ = batch["champ"].clone()
    items = batch["items"].clone()
    item_cd = batch["item_cd"].clone()
    B, T = batch["valid"].shape
    champ_mask = torch.zeros(B, T, 10, dtype=torch.bool, device=champ.device)
    for b, c, t0, t1 in spans:
        champ_mask[b, t0:t1, c] = True

    champ[champ_mask] = 0.0
    items[champ_mask] = 0
    item_cd[champ_mask] = 0.0

    out = dict(batch)
    out.update(champ=champ, items=items, item_cd=item_cd, champ_mask=champ_mask)
    return out


def sample_and_apply(batch: dict, cfg: MaskConfig,
                     generator: torch.Generator | None = None,
                     champ_slots: list[int] | None = None) -> dict:
    """Random spans over random champions (or restricted to champ_slots)."""
    valid = batch["valid"]
    B, _ = valid.shape

    def rint(lo: int, hi: int) -> int:              # inclusive bounds
        return int(torch.randint(lo, hi + 1, (1,), generator=generator))

    spans = []
    for b in range(B):
        T_g = int(valid[b].sum())
        if T_g <= cfg.span_min + 1:
            continue
        if champ_slots is None:
            champs = torch.randperm(10, generator=generator)[: cfg.n_champs].tolist()
        else:
            pick = torch.randperm(len(champ_slots), generator=generator)[: cfg.n_champs]
            champs = [champ_slots[i] for i in pick.tolist()]
        for c in champs:
            for _ in range(rint(1, cfg.spans_per_champ_max)):
                span = rint(cfg.span_min, min(cfg.span_max, T_g - 1))
                t0 = rint(0, T_g - span)
                spans.append((b, c, t0, t0 + span))
    return apply_spans(batch, spans)
