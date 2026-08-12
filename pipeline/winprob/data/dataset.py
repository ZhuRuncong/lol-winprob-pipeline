"""Full-game dataset over packed bundles, with augmentation and
token-budget length-bucketed batching.

Augmentation contract: SSL target tensors (tgt_*, pos_bin) are NEVER
jittered or dropped — they only undergo the team-relabeling swap (which is a
relabeling of the same physical game, y flips with it). Input corruption
(group dropout, jitter, channel dropout) touches inputs only.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from ..config import CI, TrainConfig
from .transforms import OFF_FOURIER, OFF_KDA, OFF_MICRO, fourier_positions

SWAP_PERM = np.r_[5:10, 0:5]

# input feature-group dropout: name -> (array key, index list into last dim)
_POS_DIMS = [CI["pos_x"], CI["pos_z"]] + list(range(OFF_FOURIER, OFF_KDA)) \
    + [OFF_MICRO + 9, OFF_MICRO + 10]                      # + displacement deltas
_CD_DIMS = [CI[n] for n in ("cd_q", "cd_w", "cd_e", "cd_r", "cd_summ1", "cd_summ2")]
_GOLD_DIMS = [CI["current_gold"], CI["total_gold"], CI["xp"], CI["cs"]]
GROUPS = {
    "positions": ("champ", _POS_DIMS),
    "cooldowns": ("champ", _CD_DIMS),
    "kda": ("champ", list(range(OFF_KDA, OFF_MICRO))),
    "micro": ("champ", list(range(OFF_MICRO, OFF_MICRO + 12))),
    "wards": ("team", list(range(26, 46))),
    "items": ("items", None),                              # zero whole arrays
}


class GameBundleDataset(Dataset):
    def __init__(self, bundles_dir: str | Path, split: str,
                 train_cfg: TrainConfig | None = None, seed: int = 0):
        self.dir = Path(bundles_dir)
        with open(self.dir / "bundles_manifest.json", encoding="utf-8") as fh:
            self.manifest = json.load(fh)
        self.rows = [g for g in self.manifest["games"] if g["split"] == split]
        if not self.rows:
            raise ValueError(f"no games for split '{split}' in {self.dir}")
        self.split = split
        self.cfg = train_cfg          # None -> eval mode, no augmentation
        self.seed = seed
        self.epoch = 0

    def __len__(self):
        return len(self.rows)

    def lengths(self) -> np.ndarray:
        return np.array([g["T"] for g in self.rows])

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def __getitem__(self, idx: int) -> dict:
        row = self.rows[idx]
        with np.load(self.dir / row["split"] / f"{row['game']}.npz") as z:
            d = {k: z[k].copy() for k in z.files}

        d["side"] = np.repeat(np.array([0, 1], dtype=np.int8), 5)
        d["swap"] = np.int8(0)
        d["game_idx"] = np.int32(idx)

        if self.cfg is not None:
            rng = np.random.default_rng(
                (self.seed * 1_000_003 + self.epoch) * 1_000_003 + idx)
            self._augment(d, rng)
        return d

    # -- augmentation ------------------------------------------------------

    def _augment(self, d: dict, rng: np.random.Generator):
        cfg = self.cfg
        if rng.random() < cfg.swap_prob:
            self._swap(d)

        champ = d["champ"].astype(np.float32)

        for name, (key, dims) in GROUPS.items():
            if rng.random() < cfg.group_dropout:
                if name == "items":
                    d["items"][:] = 0
                    d["item_cd"][:] = 0
                elif key == "champ":
                    champ[..., dims] = 0.0
                else:
                    d[key][..., dims] = 0

        if rng.random() < cfg.champ_id_dropout:
            d["statics"][:, 0] = 0                     # champion_id 0 = unknown

        if cfg.channel_dropout > 0:
            drop = rng.random(champ.shape[-1]) < cfg.channel_dropout
            champ[..., drop] = 0.0

        if cfg.jitter_pos > 0:
            pos = champ[..., [CI["pos_x"], CI["pos_z"]]]
            pos = np.clip(pos + rng.normal(0, cfg.jitter_pos, pos.shape).astype(np.float32), 0, 1)
            champ[..., CI["pos_x"]] = pos[..., 0]
            champ[..., CI["pos_z"]] = pos[..., 1]
            champ[..., OFF_FOURIER:OFF_KDA] = fourier_positions(pos)
        if cfg.jitter_gold > 0:
            champ[..., _GOLD_DIMS] += rng.normal(
                0, cfg.jitter_gold, champ[..., _GOLD_DIMS].shape).astype(np.float32)

        d["champ"] = champ.astype(np.float16)

    @staticmethod
    def _swap(d: dict):
        """Team-relabeling swap: pure relabeling, label flips, no geometry change."""
        d["champ"] = d["champ"][:, SWAP_PERM]
        d["items"] = d["items"][:, SWAP_PERM]
        d["item_cd"] = d["item_cd"][:, SWAP_PERM]
        d["statics"] = d["statics"][SWAP_PERM]
        d["tgt_champ"] = d["tgt_champ"][:, SWAP_PERM]
        d["tgt_champ_deaths"] = d["tgt_champ_deaths"][:, SWAP_PERM]
        d["pos_bin"] = d["pos_bin"][:, SWAP_PERM]
        d["team"] = d["team"][:, ::-1]
        d["tgt_team"] = d["tgt_team"][:, ::-1]
        d["side"] = d["side"][SWAP_PERM] ^ 1
        d["y"] = np.int8(1 - int(d["y"]))
        d["swap"] = np.int8(1 - int(d["swap"]))    # toggle: swap twice = identity


class TokenBudgetBatchSampler(Sampler):
    """Length-bucketed batches: sum of right-padded timesteps <= budget."""

    def __init__(self, lengths: np.ndarray, budget: int, shuffle: bool = True,
                 seed: int = 0, bucket: int = 256):
        self.lengths = np.asarray(lengths)
        self.budget = int(budget)
        self.shuffle = shuffle
        self.seed = seed
        self.bucket = bucket
        self.epoch = 0
        if self.lengths.max() > self.budget:
            raise ValueError(f"budget {budget} < longest game {self.lengths.max()}")

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def _batches(self) -> list[list[int]]:
        idx = np.arange(len(self.lengths))
        rng = np.random.default_rng(self.seed * 100_003 + self.epoch)
        if self.shuffle:
            rng.shuffle(idx)
        keys = (self.lengths[idx] // self.bucket)
        idx = idx[np.argsort(keys, kind="stable")]

        batches, cur, cur_max = [], [], 0
        for i in idx:
            t = int(self.lengths[i])
            new_max = max(cur_max, t)
            if cur and new_max * (len(cur) + 1) > self.budget:
                batches.append(cur)
                cur, cur_max = [], 0
                new_max = t
            cur.append(int(i))
            cur_max = new_max
        if cur:
            batches.append(cur)
        if self.shuffle:
            rng.shuffle(batches)
        return batches

    def __iter__(self):
        return iter(self._batches())

    def __len__(self):
        return len(self._batches())


PAD_KEYS_TIME = ("champ", "team", "glob", "items", "item_cd",
                 "tgt_champ", "tgt_champ_deaths", "tgt_team", "pos_bin")


def collate(batch: list[dict]) -> dict[str, torch.Tensor]:
    """Right-pad every time-indexed array to the batch max T; emit `valid`."""
    B = len(batch)
    Ts = [b["champ"].shape[0] for b in batch]
    Tm = max(Ts)
    out: dict[str, torch.Tensor] = {}
    for k in PAD_KEYS_TIME:
        arrs = []
        for b in batch:
            a = b[k]
            pad = [(0, Tm - a.shape[0])] + [(0, 0)] * (a.ndim - 1)
            arrs.append(np.pad(a, pad))
        out[k] = torch.from_numpy(np.stack(arrs))
    out["statics"] = torch.from_numpy(np.stack([b["statics"] for b in batch]).astype(np.int64))
    out["side"] = torch.from_numpy(np.stack([b["side"] for b in batch]).astype(np.int64))
    out["patch_idx"] = torch.tensor([int(b["patch_idx"]) for b in batch], dtype=torch.int64)
    out["swap"] = torch.tensor([int(b["swap"]) for b in batch], dtype=torch.float32)
    out["y"] = torch.tensor([float(b["y"]) for b in batch], dtype=torch.float32)
    out["game_idx"] = torch.tensor([int(b["game_idx"]) for b in batch], dtype=torch.int64)
    valid = torch.zeros(B, Tm, dtype=torch.bool)
    for i, t in enumerate(Ts):
        valid[i, :t] = True
    out["valid"] = valid
    out["items"] = out["items"].to(torch.int64)
    out["pos_bin"] = out["pos_bin"].to(torch.int64)
    return out
