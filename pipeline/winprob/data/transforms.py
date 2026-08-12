"""Raw game arrays -> model feature blocks and SSL target groups.

Shared by stats.py (which streams TRAIN games to fit normalization) and
pack.py (which applies the fitted stats and writes bundles). Everything here
is pure numpy and strictly causal: no value at time t may derive from any
event after t (ward t_end is used only for the alive-at-t predicate).
"""
from __future__ import annotations

import numpy as np

from ..config import (
    CHAMP_CD_BOUND, CHAMP_LEVEL5, CHAMP_LEVEL18, CHAMP_LOG1P_Z, CHAMP_Z,
    CI, CLOCK_PERIODS, D_CHAMP_CONT, D_FOURIER, D_GLOBAL_CONT, D_KDA, D_MICRO,
    D_TEAM_CONT, D_WARD, FOURIER_K, ITEM_HASH_BUCKETS, MICRO_DISP_WINDOWS,
    MICRO_RATE_FEATURES, MICRO_RATE_WINDOWS, POS_GRID, ROLE_IDX, ROLES,
    STATIC_COLS, STYLE_VOCAB, TEAM_LOG1P_Z, TEAM_RAW, TEAM_TIMER300, TI,
    WARD_GRID, X_CHAMP_FEATURES, X_TEAM_FEATURES, item_hash, keystone_hash,
)

# ---------------------------------------------------------------------------
# Z-mask layout: which dims of each packed block get train-stat z-scoring
# ---------------------------------------------------------------------------

N_BASE_C = len(X_CHAMP_FEATURES)          # 26
OFF_FOURIER = N_BASE_C                    # 26
OFF_KDA = OFF_FOURIER + D_FOURIER         # 42
OFF_MICRO = OFF_KDA + D_KDA               # 46
assert OFF_MICRO + D_MICRO == D_CHAMP_CONT

CHAMP_Z_MASK = np.zeros(D_CHAMP_CONT, dtype=bool)
for name in CHAMP_LOG1P_Z + CHAMP_Z:
    CHAMP_Z_MASK[CI[name]] = True
CHAMP_Z_MASK[OFF_KDA:OFF_KDA + 3] = True          # kills, deaths, assists
CHAMP_Z_MASK[OFF_MICRO:OFF_MICRO + D_MICRO] = True

N_BASE_T = len(X_TEAM_FEATURES)           # 26
TEAM_Z_MASK = np.zeros(D_TEAM_CONT, dtype=bool)
for name in X_TEAM_FEATURES:
    if name not in TEAM_TIMER300 + TEAM_RAW:
        TEAM_Z_MASK[TI[name]] = True      # includes total_gold (log1p'd first)

# targets layout
TGT_CHAMP_FEATURES = ["cs", "total_gold", "xp", "hp_frac", "mana_frac", "level", "alive"]
TGT_TEAM_FEATURES = ["kills", "towers_destroyed", "dragons_cum", "baron_cum", "total_gold"]


# ---------------------------------------------------------------------------
# Blocks
# ---------------------------------------------------------------------------

def champ_cont_prenorm(X_champ: np.ndarray, X_kda: np.ndarray) -> np.ndarray:
    """[T,10,26] + [T,10,4] -> [T,10,58] pre-normalization champion block."""
    T = X_champ.shape[0]
    out = np.zeros((T, 10, D_CHAMP_CONT), dtype=np.float32)

    base = X_champ.astype(np.float32).copy()
    for name in CHAMP_LOG1P_Z:
        base[:, :, CI[name]] = np.log1p(np.maximum(base[:, :, CI[name]], 0.0))
    for name in CHAMP_CD_BOUND:
        base[:, :, CI[name]] = np.minimum(base[:, :, CI[name]], 300.0) / 60.0
    for name in CHAMP_LEVEL18:
        base[:, :, CI[name]] /= 18.0
    for name in CHAMP_LEVEL5:
        base[:, :, CI[name]] /= 5.0
    out[:, :, :N_BASE_C] = base

    pos = X_champ[:, :, [CI["pos_x"], CI["pos_z"]]].astype(np.float32)
    out[:, :, OFF_FOURIER:OFF_KDA] = fourier_positions(pos)

    kda = X_kda.astype(np.float32).copy()
    kda[:, :, 3] /= 50.0                              # vision_score
    out[:, :, OFF_KDA:OFF_MICRO] = kda

    out[:, :, OFF_MICRO:] = micro_deltas(X_champ)
    return out


def fourier_positions(pos: np.ndarray) -> np.ndarray:
    """[..., 2] in [0,1] -> [..., 16] sin/cos(2^k * pi * p), k=0..3."""
    feats = []
    for k in range(FOURIER_K):
        arg = (2.0 ** k) * np.pi * pos
        feats.append(np.sin(arg))
        feats.append(np.cos(arg))
    return np.concatenate(feats, axis=-1).astype(np.float32)


def micro_deltas(X_champ: np.ndarray) -> np.ndarray:
    """Trailing (past-only) rates/displacements. [T,10,26] -> [T,10,12].

    Windows clamp at the game start; rates divide by actual elapsed time so
    early-game values stay on scale.
    """
    T = X_champ.shape[0]
    out = np.zeros((T, 10, D_MICRO), dtype=np.float32)
    ts = np.arange(T)
    col = 0
    for w in MICRO_RATE_WINDOWS:
        back = np.maximum(ts - w, 0)
        elapsed = np.maximum(ts - back, 1).astype(np.float32)[:, None]
        for name in MICRO_RATE_FEATURES:
            x = X_champ[:, :, CI[name]]
            out[:, :, col] = (x - x[back]) / elapsed
            col += 1
    pos = X_champ[:, :, [CI["pos_x"], CI["pos_z"]]]
    for w in MICRO_DISP_WINDOWS:
        back = np.maximum(ts - w, 0)
        out[:, :, col] = np.linalg.norm(pos - pos[back], axis=-1)
        col += 1
    back5 = np.maximum(ts - 5, 0)
    hp = X_champ[:, :, CI["hp_frac"]]
    out[:, :, col] = hp - hp[back5]
    return out


def team_cont_prenorm(X_team: np.ndarray, wards: np.ndarray, T: int) -> np.ndarray:
    """[T,2,26] + ward table -> [T,2,46] pre-normalization team block."""
    out = np.zeros((T, 2, D_TEAM_CONT), dtype=np.float32)
    base = X_team.astype(np.float32).copy()
    for name in TEAM_LOG1P_Z:
        base[:, :, TI[name]] = np.log1p(np.maximum(base[:, :, TI[name]], 0.0))
    for name in TEAM_TIMER300:
        base[:, :, TI[name]] /= 300.0
    out[:, :, :N_BASE_T] = base
    out[:, :, N_BASE_T:] = np.log1p(ward_raster(wards, T))
    return out


def ward_raster(wards: np.ndarray, T: int) -> np.ndarray:
    """Interval table [W,6] -> per-second per-team alive counts [T,2,20].

    Channels: 16 = 4x4 map-grid cells, then 4 = ward-type counts. Uses only
    the alive-at-t predicate (t_start <= t < t_end); no t_end-derived values.
    """
    out = np.zeros((T + 1, 2, D_WARD), dtype=np.float32)
    for team, wtype, x, z, ts, te in np.asarray(wards, dtype=np.float64):
        a = int(np.clip(np.ceil(ts), 0, T))
        b = int(np.clip(np.ceil(te), 0, T))
        if b <= a:
            continue
        tm = int(team)
        gx = min(int(x * WARD_GRID), WARD_GRID - 1)
        gz = min(int(z * WARD_GRID), WARD_GRID - 1)
        out[a, tm, gx * WARD_GRID + gz] += 1
        out[b, tm, gx * WARD_GRID + gz] -= 1
        tc = WARD_GRID * WARD_GRID + int(np.clip(wtype, 0, 3))
        out[a, tm, tc] += 1
        out[b, tm, tc] -= 1
    return np.cumsum(out[:T], axis=0)


def global_cont(X_global: np.ndarray, T: int) -> np.ndarray:
    """[T,6] -> [T,12]: transformed globals + absolute-clock Fourier features."""
    out = np.zeros((T, D_GLOBAL_CONT), dtype=np.float32)
    g = X_global.astype(np.float32).copy()
    g[:, 1] = np.minimum(g[:, 1], 600.0) / 60.0       # next_dragon_spawn_in
    g[:, 2] = np.minimum(g[:, 2], 600.0) / 60.0       # next_baron_or_herald
    out[:, :6] = g
    ts = np.arange(T, dtype=np.float32)
    for i, period in enumerate(CLOCK_PERIODS):
        arg = 2.0 * np.pi * ts / period
        out[:, 6 + 2 * i] = np.sin(arg)
        out[:, 6 + 2 * i + 1] = np.cos(arg)
    return out


# ---------------------------------------------------------------------------
# Items / statics / targets
# ---------------------------------------------------------------------------

_ITEM_LUT: dict[int, np.ndarray] = {}


def hash_items(X_items: np.ndarray) -> np.ndarray:
    """[T,10,7] raw Riot item ids -> int16 hash-bucket indices."""
    ids = np.unique(X_items)
    lut = {int(i): item_hash(int(i)) for i in ids}
    return np.vectorize(lut.get, otypes=[np.int16])(X_items)


def item_cd_feats(X_item_cd: np.ndarray) -> np.ndarray:
    return (np.minimum(X_item_cd.astype(np.float32), 300.0) / 60.0)


def role_order(participants: list[dict]) -> tuple[np.ndarray, bool]:
    """Permutation mapping canonical slot -> original slot index.

    Canonical: [blue Top,Jg,Mid,Bot,Sup, red Top,Jg,Mid,Bot,Sup]. Unmatched
    roles fall back to participant order within the team; ok=False flags it.
    """
    perm = np.full(10, -1, dtype=np.int64)
    ok = True
    for side, team_id in enumerate((100, 200)):
        members = [i for i, p in enumerate(participants) if p.get("team") == team_id]
        used = set()
        for r, role in enumerate(ROLES):
            for i in members:
                if i in used:
                    continue
                if str(participants[i].get("role") or "").lower() == role.lower():
                    perm[side * 5 + r] = i
                    used.add(i)
                    break
        leftovers = [i for i in members if i not in used]
        for slot in range(side * 5, side * 5 + 5):
            if perm[slot] == -1:
                perm[slot] = leftovers.pop(0)
                ok = False
        if len(members) != 5:
            ok = False
    return perm, ok


def champ_statics(participants: list[dict]) -> np.ndarray:
    """[10, 7] int16: champion_id, role_idx, spell1, spell2, keystone, styles."""
    out = np.zeros((10, len(STATIC_COLS)), dtype=np.int16)
    for i, p in enumerate(participants[:10]):
        champ = p.get("champion_id") or 0
        role = ROLE_IDX.get(str(p.get("role") or "").lower(), i % 5)
        spells = [(s or 0) for s in (p.get("summoner_spells") or [0, 0])[:2]]
        while len(spells) < 2:
            spells.append(0)
        runes = p.get("runes") or {}
        ks = runes.get("keystone") or 0
        style = runes.get("style")
        sub = runes.get("sub_style")

        def style_idx(s):
            if not s:
                return STYLE_VOCAB - 1
            return int(np.clip((int(s) - 8000) // 100, 0, 4))

        out[i] = [
            int(np.clip(champ, 0, 1023)),
            role,
            int(np.clip(spells[0], 0, 63)),
            int(np.clip(spells[1], 0, 63)),
            keystone_hash(int(ks)) if ks else 0,
            style_idx(style),
            style_idx(sub),
        ]
    return out


def pos_bins(X_champ: np.ndarray) -> np.ndarray:
    """[T,10,26] -> int16 [T,10] position bin on the POS_GRID x POS_GRID grid."""
    x = np.clip((X_champ[:, :, CI["pos_x"]] * POS_GRID).astype(np.int64), 0, POS_GRID - 1)
    z = np.clip((X_champ[:, :, CI["pos_z"]] * POS_GRID).astype(np.int64), 0, POS_GRID - 1)
    return (x * POS_GRID + z).astype(np.int16)


def build_targets(X_champ, X_kda, X_team) -> dict[str, np.ndarray]:
    """Raw-value target group for SSL objectives (always from clean tensors)."""
    tgt_champ = np.stack(
        [X_champ[:, :, CI[n]] for n in ("cs", "total_gold", "xp", "hp_frac",
                                        "mana_frac", "level", "alive")],
        axis=-1).astype(np.float32)

    dragons = X_team[:, :, [TI[f"dragons_{d}"] for d in
                            ("fire", "water", "earth", "air", "hextech", "chemtech")]].sum(-1)
    # elder / baron takes are visible only as buff-flag rising edges
    elder_edge = np.diff(X_team[:, :, TI["elder_active"]], axis=0, prepend=0.0) > 0.5
    baron_edge = np.diff(X_team[:, :, TI["baron_active"]], axis=0, prepend=0.0) > 0.5
    tgt_team = np.stack([
        X_team[:, :, TI["kills"]],
        X_team[:, :, TI["towers_destroyed"]],
        dragons + np.cumsum(elder_edge, axis=0),
        np.cumsum(baron_edge, axis=0).astype(np.float32),
        X_team[:, :, TI["total_gold"]],
    ], axis=-1).astype(np.float32)

    return {
        "tgt_champ": tgt_champ,                         # [T,10,7]
        "tgt_champ_deaths": X_kda[:, :, 1].astype(np.float32),  # [T,10] cumulative
        "tgt_team": tgt_team,                           # [T,2,5]
        "pos_bin": pos_bins(X_champ),                   # [T,10]
    }


# ---------------------------------------------------------------------------
# Normalization application
# ---------------------------------------------------------------------------

def apply_z(block: np.ndarray, mask: np.ndarray, mean: np.ndarray, std: np.ndarray,
            clip: float = 8.0) -> np.ndarray:
    """Z-score the masked dims in place (mean/std indexed over masked dims only)."""
    sub = (block[..., mask] - mean) / std
    block[..., mask] = np.clip(sub, -clip, clip)
    return block


def parse_patch(patch: str | None) -> str:
    if not patch:
        return "?"
    parts = str(patch).split(".")
    return ".".join(parts[:2]) if len(parts) >= 2 else parts[0]
