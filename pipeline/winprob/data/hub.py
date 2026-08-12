"""Corpus acquisition: pinned HF snapshot or the local 142-game smoke sample.

The corpus root is a directory with manifest.json, features.json, and
shards/*.tar (each tar member: <game>/{meta.json, sequence.npz}) — the layout
`pipeline pack` publishes. On first contact the feature orders are verified
against the contract in config.py; a mismatch is a hard error, never a
silent reinterpretation.
"""
from __future__ import annotations

import io
import json
import tarfile
from pathlib import Path

import numpy as np

from ..config import (HF_REPO, HF_REVISION, X_CHAMP_FEATURES, X_GLOBAL_FEATURES,
                      X_TEAM_FEATURES)

REPO_ROOT = Path(__file__).resolve().parents[3]
LOCAL_SAMPLE = REPO_ROOT / "data" / "pack"


def corpus_root(local_sample: bool = False, repo: str = HF_REPO,
                revision: str = HF_REVISION) -> Path:
    if local_sample:
        root = LOCAL_SAMPLE
        if not (root / "manifest.json").exists():
            raise SystemExit(f"local sample not found: {root} (run the pack stage first)")
    else:
        from huggingface_hub import snapshot_download
        root = Path(snapshot_download(repo, revision=revision, repo_type="dataset"))
    verify_features(root)
    return root


def verify_features(root: Path) -> None:
    with open(root / "features.json", encoding="utf-8") as fh:
        feats = json.load(fh)
    for key, expected in (("X_champ", X_CHAMP_FEATURES),
                          ("X_team", X_TEAM_FEATURES),
                          ("X_global", X_GLOBAL_FEATURES)):
        if feats.get(key) != expected:
            raise SystemExit(
                f"feature contract mismatch for {key} in {root}/features.json — "
                f"config.py expects {expected}, corpus has {feats.get(key)}")


def iter_shard_games(root: Path, wanted: set[str] | None = None):
    """Yield (game_name, meta_dict, npz_dict) from shards/*.tar.

    `wanted` filters by game name (e.g. only train games) without extracting
    the rest.
    """
    for tar_path in sorted((root / "shards").glob("*.tar")):
        with tarfile.open(tar_path) as tf:
            groups: dict[str, dict[str, tarfile.TarInfo]] = {}
            for m in tf.getmembers():
                if not m.isfile():
                    continue
                parts = m.name.split("/")
                if len(parts) >= 2:
                    groups.setdefault(parts[0], {})[parts[-1]] = m
            for game, files in sorted(groups.items()):
                if wanted is not None and game not in wanted:
                    continue
                if "meta.json" not in files or "sequence.npz" not in files:
                    continue
                meta = json.loads(tf.extractfile(files["meta.json"]).read().decode("utf-8"))
                with np.load(io.BytesIO(tf.extractfile(files["sequence.npz"]).read())) as z:
                    npz = {k: z[k] for k in z.files}
                yield game, meta, npz
