"""Central configuration: the feature contract, model sizes, and run settings.

Every module imports its dimensions from here. If a number describes the shape
of packed data or a model input, it lives in this file and nowhere else.
"""
from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass, replace
from pathlib import Path

# ---------------------------------------------------------------------------
# Corpus
# ---------------------------------------------------------------------------

HF_REPO = "zauberine/lol-winprob"
# Pinned: training data is this revision, always.
HF_REVISION = "0b642533409636fae5b98d2e4f6646f73b2075f2"

# ---------------------------------------------------------------------------
# Raw feature orders (must match data/pack features.json — verified at pack)
# ---------------------------------------------------------------------------

X_CHAMP_FEATURES = [
    "alive", "respawn_timer", "level", "xp", "current_gold", "total_gold",
    "hp", "hp_max", "hp_frac", "mana", "mana_max", "mana_frac",
    "cd_q", "cd_w", "cd_e", "cd_r", "lvl_q", "lvl_w", "lvl_e", "lvl_r",
    "cd_summ1", "cd_summ2", "pos_x", "pos_z", "shutdown_value", "cs",
]
X_TEAM_FEATURES = [
    "kills", "deaths", "assists", "total_gold",
    "towers_destroyed", "inhibs_destroyed",
    "turrets_alive_top", "turrets_alive_mid", "turrets_alive_bot", "nexus_turrets_alive",
    "plates_remaining_top", "plates_remaining_mid", "plates_remaining_bot",
    "inhib_down_timer_top", "inhib_down_timer_mid", "inhib_down_timer_bot",
    "dragons_fire", "dragons_water", "dragons_earth", "dragons_air",
    "dragons_hextech", "dragons_chemtech",
    "elder_active", "baron_active", "grubs_taken", "herald_taken",
]
X_GLOBAL_FEATURES = [
    "game_time_norm", "next_dragon_spawn_in", "next_baron_or_herald_spawn_in",
    "is_pre_plates", "dragon_up", "baron_or_herald_up",
]

CI = {name: i for i, name in enumerate(X_CHAMP_FEATURES)}   # champ index
TI = {name: i for i, name in enumerate(X_TEAM_FEATURES)}    # team index

ROLES = ["Top", "Jungle", "Middle", "Bottom", "Support"]  # feed's exact vocabulary
ROLE_IDX = {r.lower(): i for i, r in enumerate(ROLES)}
ROLE_IDX["mid"] = ROLE_IDX["middle"]                      # tolerated alias

# ---------------------------------------------------------------------------
# Transform classes (applied in data/transforms.py; z-stats from TRAIN only)
# ---------------------------------------------------------------------------

CHAMP_LOG1P_Z = ["xp", "current_gold", "total_gold", "shutdown_value", "cs"]
CHAMP_CD_BOUND = ["respawn_timer", "cd_q", "cd_w", "cd_e", "cd_r",
                  "cd_summ1", "cd_summ2"]          # min(x, 300) / 60
CHAMP_RAW = ["alive", "hp_frac", "mana_frac", "pos_x", "pos_z"]
CHAMP_LEVEL18 = ["level"]                          # / 18
CHAMP_LEVEL5 = ["lvl_q", "lvl_w", "lvl_e", "lvl_r"]  # / 5
CHAMP_Z = ["hp", "hp_max", "mana", "mana_max"]

TEAM_TIMER300 = ["inhib_down_timer_top", "inhib_down_timer_mid", "inhib_down_timer_bot"]
TEAM_RAW = ["elder_active", "baron_active"]
TEAM_LOG1P_Z = ["total_gold"]
TEAM_Z = [f for f in X_TEAM_FEATURES
          if f not in TEAM_TIMER300 + TEAM_RAW + TEAM_LOG1P_Z]

# Fourier position lifting: sin/cos(2^k * pi * p) for k in 0..3, both axes.
FOURIER_K = 4
D_FOURIER = 2 * 2 * FOURIER_K  # 16

# Causal micro-deltas (trailing, past-only): rates over {30,60,90}s for
# {cs, total_gold, xp}; displacement norm over {10,30}s; hp_frac delta over 5s.
MICRO_RATE_WINDOWS = [30, 60, 90]
MICRO_RATE_FEATURES = ["cs", "total_gold", "xp"]
MICRO_DISP_WINDOWS = [10, 30]
D_MICRO = len(MICRO_RATE_WINDOWS) * len(MICRO_RATE_FEATURES) + len(MICRO_DISP_WINDOWS) + 1  # 12

D_KDA = 4  # kills, deaths, assists (z), vision_score / 50

# Packed per-champion continuous vector: 26 base + 16 fourier + 4 kda + 12 micro
D_CHAMP_CONT = len(X_CHAMP_FEATURES) + D_FOURIER + D_KDA + D_MICRO  # 58

# Ward raster: per team, 4x4 grid alive-counts + 4 type counts, log1p'd.
WARD_GRID = 4
D_WARD = WARD_GRID * WARD_GRID + 4  # 20

# Packed team vector: 26 transformed + 20 ward raster
D_TEAM_CONT = len(X_TEAM_FEATURES) + D_WARD  # 46

# Packed global vector: 6 transformed + 6 clock fourier
CLOCK_PERIODS = [60.0, 300.0, 1800.0]
D_GLOBAL_CONT = len(X_GLOBAL_FEATURES) + 2 * len(CLOCK_PERIODS)  # 12

# ---------------------------------------------------------------------------
# Categorical vocabularies
# ---------------------------------------------------------------------------

ITEM_HASH_BUCKETS = 4096      # bucket 0 reserved for "empty slot"
N_ITEM_SLOTS = 7
CHAMPION_VOCAB = 1024         # direct Riot champion_id lookup (ids < 1024)
SPELL_VOCAB = 64              # direct summoner-spell id lookup (clamped)
KEYSTONE_HASH_BUCKETS = 64
STYLE_VOCAB = 8               # (rune_style - 8000) // 100, clamped; OOV -> 7
PATCH_OOV = 0                 # patch vocab built from train split; 0 = OOV

# statics_champ int16 columns
STATIC_COLS = ["champion_id", "role_idx", "spell1", "spell2",
               "keystone_hash", "style_idx", "substyle_idx"]

# Position grid for SSL targets (24x24 = 576-way classification)
POS_GRID = 24
POS_BINS = POS_GRID * POS_GRID


def item_hash(item_id: int) -> int:
    """Stable hash of a Riot item id into [1, ITEM_HASH_BUCKETS); 0 stays 0."""
    if item_id == 0:
        return 0
    return 1 + (item_id * 2654435761) % (ITEM_HASH_BUCKETS - 1)


def keystone_hash(rune_id: int) -> int:
    return (rune_id * 2654435761) % KEYSTONE_HASH_BUCKETS


# ---------------------------------------------------------------------------
# SSL objective spec
# ---------------------------------------------------------------------------

DELTA_HORIZONS = [5, 30, 120]        # future-delta horizons (s)
DELTA_CHAMP_FEATURES = ["total_gold", "xp", "cs"]
DELTA_TEAM_FEATURES = ["gold_diff", "towers_destroyed"]
POS_HORIZONS = [15, 60]              # future-position horizons (s)
HAZARDS = [                          # (name, scope, horizon_s)
    ("death", "champ", 30),
    ("kill", "team", 15),
    ("tower", "team", 60),
    ("dragon", "team", 60),
    ("baron", "team", 60),
]

# ---------------------------------------------------------------------------
# Dataclass configs
# ---------------------------------------------------------------------------


@dataclass
class ModelConfig:
    d_tok: int = 128
    set_layers: int = 2
    set_heads: int = 4
    d_model: int = 384
    layers: int = 8
    heads: int = 12
    mlp_ratio: int = 4
    dropout: float = 0.1
    stoch_depth: float = 0.1
    win_head_hidden: int = 64
    win_head_dropout: float = 0.3
    d_champ_emb: int = 32
    d_role_emb: int = 8
    d_spell_emb: int = 8
    d_keystone_emb: int = 8
    d_style_emb: int = 2
    d_side_emb: int = 2
    d_patch_emb: int = 8
    d_item_emb: int = 16
    rope_theta: float = 10000.0
    # Ablation flags (never forks): 0 = full causal context.
    attn_window: int = 0          # causal attention window in seconds
    context_one: bool = False     # per-second MLP: attend only to self


@dataclass
class MaskConfig:
    n_champs: int = 2             # champions masked per sequence
    spans_per_champ_max: int = 3
    span_min: int = 15
    span_max: int = 45


@dataclass
class LossWeights:
    mask: float = 1.0
    delta: float = 1.0
    pos: float = 0.5
    hazard: float = 0.5
    tte: float = 0.25             # time-to-end (phase A only)
    win: float = 1.0              # phase B
    ssl_retention: float = 0.25   # phase B multiplier on delta/pos/hazard


@dataclass
class TrainConfig:
    phase: str = "ssl"            # "ssl" | "finetune"
    token_budget: int = 65536     # timesteps per batch (length-bucketed)
    lr: float = 3e-4              # phase A lr / phase B head lr
    backbone_lr: float = 5e-5     # phase B only
    weight_decay: float = 0.05
    betas: tuple = (0.9, 0.95)
    grad_clip: float = 1.0
    warmup_frac: float = 0.03
    epochs: int = 60
    ema_decay: float = 0.999
    seed: int = 1
    amp_dtype: str = "bf16"       # "bf16" | "fp16" | "fp32"
    win_stride: int = 5           # phase B win-BCE subsampling stride
    quarter_balance: bool = True  # phase B within-game quarter balancing
    # Augmentation / regularization
    swap_prob: float = 0.5
    group_dropout: float = 0.07   # per feature group per game
    champ_id_dropout: float = 0.1
    channel_dropout: float = 0.05
    jitter_pos: float = 0.007
    jitter_gold: float = 0.005
    # Early stopping (phase B: val 10-25 min bucket log-loss)
    patience: int = 5
    num_workers: int = 8


@dataclass
class DataConfig:
    bundles_dir: str = "data/winprob_bundles"
    local_sample: bool = False    # use data/processed instead of the HF snapshot
    hf_repo: str = HF_REPO
    hf_revision: str = HF_REVISION


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    mask: MaskConfig = field(default_factory=MaskConfig)
    loss: LossWeights = field(default_factory=LossWeights)
    train: TrainConfig = field(default_factory=TrainConfig)
    data: DataConfig = field(default_factory=DataConfig)
    name: str = "default"


TRUNK_PRESETS = {
    "S": dict(d_model=256, layers=6, heads=8),
    "M": dict(d_model=384, layers=8, heads=12),
    "L": dict(d_model=512, layers=12, heads=16),
    "smoke": dict(d_model=64, layers=2, heads=2, d_tok=32, set_layers=1),
}


def load_config(path: str | Path | None = None, **overrides) -> Config:
    """Config from optional YAML + keyword overrides (dotted keys allowed)."""
    cfg = Config()
    merged: dict = {}
    if path is not None:
        import yaml
        with open(path, "r", encoding="utf-8") as fh:
            merged.update(yaml.safe_load(fh) or {})
    merged.update(overrides)

    if "trunk" in merged:  # e.g. trunk: M
        preset = TRUNK_PRESETS[merged.pop("trunk")]
        for k, v in preset.items():
            merged.setdefault("model", {})
            if isinstance(merged["model"], dict):
                merged["model"].setdefault(k, v)

    def apply(dc, values: dict):
        kw = {}
        for f in fields(dc):
            if f.name not in values:
                continue
            v = values[f.name]
            cur = getattr(dc, f.name)
            kw[f.name] = apply(cur, v) if is_dataclass(cur) and isinstance(v, dict) else v
        return replace(dc, **kw)

    for key, value in merged.items():
        if key in ("model", "mask", "loss", "train", "data") and isinstance(value, dict):
            cfg = replace(cfg, **{key: apply(getattr(cfg, key), value)})
        elif "." in key:
            section, leaf = key.split(".", 1)
            cfg = replace(cfg, **{section: replace(getattr(cfg, section), **{leaf: value})})
        else:
            cfg = replace(cfg, **{key: value})
    return cfg
