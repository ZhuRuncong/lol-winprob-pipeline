"""Win-probability model: p(blue wins | game state up to t) at every second.

Two-phase training on the published corpus (HF: zauberine/lol-winprob):
  Phase A — label-free self-supervised pretraining (masked fog-of-war
            reconstruction, multi-horizon future deltas, future position,
            event hazards) fits the causal transformer trunk on ~12M dense
            per-second targets.
  Phase B — a small win head + low-LR finetune consumes the ~6.2k game labels.

The package treats GAMES, not timesteps, as the unit of supervision
everywhere: splits, loss averaging, bootstrap, calibration.
"""
