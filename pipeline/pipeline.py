"""End-to-end pipeline: GRID API -> processed per-second tensors -> Hugging Face dataset.

Publishes ONLY processed derivatives (meta.json + sequence.npz). Raw event feeds are
downloaded to a temp dir, processed, and deleted immediately. Scope is restricted to
NON-SCRIM (ESPORTS) series from the past N years (default 5).

Keys / environment variables (read from a .env file in the project root if present,
process environment always wins over .env):
  GRID_API_KEY     required for fetch/ingest   GRID API key
  HF_TOKEN         required for upload         Hugging Face write token
  HF_DATASET_REPO  required for upload         e.g. "youruser/lol-winprob"
  HF_PRIVATE       optional (default "0")      "1" -> create the HF repo as private
  GRID_TITLE_ID    optional (default "3")      GRID title id for League of Legends
  YEARS_BACK       optional (default "5")      series start-time window
  WORK_DIR         optional                    defaults to <project>/data

Stages (resumable; state kept in WORK_DIR/pipeline_state.json):
  python -m pipeline.pipeline fetch            # list eligible series ids from GRID
  python -m pipeline.pipeline ingest [--limit N]   # download -> process -> delete raw
  python -m pipeline.pipeline pack [--strict]  # shards + manifest + norm into WORK_DIR/pack
  python -m pipeline.pipeline upload           # push pack/ to the HF dataset repo
  python -m pipeline.pipeline all
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import os
import re
import shutil
import sys
import tarfile
import tempfile
import time
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_processed import (  # noqa: E402
    process_game,
    write_features_json,
    write_norm_json,
)

ROOT = Path(__file__).resolve().parent.parent


def load_dotenv(path: Path) -> None:
    """Minimal .env loader (no dependency). Process env always takes precedence."""
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key, val = key.strip(), val.strip().strip('"').strip("'")
        os.environ.setdefault(key, val)


load_dotenv(ROOT / ".env")

WORK_DIR = Path(os.environ.get("WORK_DIR", ROOT / "data"))
PROCESSED_DIR = WORK_DIR / "processed"
PACK_DIR = WORK_DIR / "pack"
STATE_PATH = WORK_DIR / "pipeline_state.json"

GRID_GRAPHQL_URL = "https://api.grid.gg/central-data/graphql"
GRID_FILE_LIST_URL = "https://api.grid.gg/file-download/list/{series_id}"

SCHEMA_VERSION = "v1"
GAMES_PER_SHARD = 100
REQUEST_GAP_S = 3.0           # GRID limit is 40 req / 60s window; 3.0s = 20 req/min,
                              # i.e. 50% of the allowed rate (measured via x-ratelimit-* headers)
SPLIT_PCT = {"train": 80, "val": 10, "test": 10}  # assigned by series-id hash

SERIES_QUERY = """
query Eligible($first: Int!, $after: Cursor, $gte: String!, $lte: String!, $titleId: ID!) {
  allSeries(
    first: $first
    after: $after
    filter: {
      titleId: $titleId
      types: ESPORTS
      startTimeScheduled: { gte: $gte, lte: $lte }
    }
    orderBy: StartTimeScheduled
  ) {
    totalCount
    pageInfo { hasNextPage endCursor }
    edges { node { id startTimeScheduled } }
  }
}
"""


# ---------------------------------------------------------------- helpers

def require_env(name: str) -> str:
    v = os.environ.get(name)
    if not v:
        sys.exit(f"missing required environment variable: {name}")
    return v


def load_state() -> dict:
    if STATE_PATH.exists():
        with open(STATE_PATH, encoding="utf-8") as fh:
            return json.load(fh)
    return {"series": {}}


def save_state(state: dict) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = STATE_PATH.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(state, fh, indent=1)
    tmp.replace(STATE_PATH)


def api_request(method: str, url: str, api_key: str, *, json_body=None, stream=False,
                max_tries: int = 5) -> requests.Response:
    """GRID request with 429/5xx exponential backoff."""
    headers = {"x-api-key": api_key}
    for attempt in range(max_tries):
        time.sleep(REQUEST_GAP_S)
        resp = requests.request(method, url, headers=headers, json=json_body,
                                stream=stream, timeout=300)
        if resp.status_code == 429 or resp.status_code >= 500:
            wait = 2 ** attempt * 2
            print(f"  {resp.status_code} from API, retrying in {wait}s", flush=True)
            time.sleep(wait)
            continue
        resp.raise_for_status()
        return resp
    raise RuntimeError(f"giving up on {url} after {max_tries} tries")


def split_for(series_id: str) -> str:
    """Deterministic train/val/test split hashed at SERIES level (no series leakage)."""
    h = int(hashlib.md5(str(series_id).encode()).hexdigest(), 16) % 100
    if h < SPLIT_PCT["train"]:
        return "train"
    if h < SPLIT_PCT["train"] + SPLIT_PCT["val"]:
        return "val"
    return "test"


# ---------------------------------------------------------------- stage: fetch

def stage_fetch(state: dict) -> None:
    """List eligible series: this title, ESPORTS (non-scrim) only, past N years."""
    api_key = require_env("GRID_API_KEY")
    title_id = os.environ.get("GRID_TITLE_ID", "3")
    years = int(os.environ.get("YEARS_BACK", "5"))
    now = datetime.now(timezone.utc)
    gte = (now - timedelta(days=365 * years)).strftime("%Y-%m-%dT%H:%M:%SZ")
    lte = now.strftime("%Y-%m-%dT%H:%M:%SZ")
    print(f"fetching ESPORTS series, title {title_id}, window {gte} .. {lte}")

    after, total, added = None, None, 0
    while True:
        resp = api_request("POST", GRID_GRAPHQL_URL, api_key, json_body={
            "query": SERIES_QUERY,
            "variables": {"first": 50, "after": after, "gte": gte, "lte": lte,
                          "titleId": title_id},
        })
        payload = resp.json()
        if "errors" in payload:
            sys.exit(f"GraphQL errors (query shape may need adjusting for your key's "
                     f"schema version): {payload['errors']}")
        conn = payload["data"]["allSeries"]
        total = conn["totalCount"]
        for edge in conn["edges"]:
            node = edge["node"]
            sid = str(node["id"])
            entry = state["series"].setdefault(sid, {})
            entry.setdefault("status", "pending")
            entry["start_time"] = node.get("startTimeScheduled")
            entry["eligible"] = True  # matched the ESPORTS + window filter
            added += 1
        print(f"  {added}/{total} series listed", flush=True)
        save_state(state)
        if not conn["pageInfo"]["hasNextPage"]:
            break
        after = conn["pageInfo"]["endCursor"]
    print(f"fetch done: {added} eligible series in state")


# ---------------------------------------------------------------- stage: ingest

def decompress_if_needed(path: Path) -> Path:
    """Events downloads may be zip- or gzip-wrapped jsonl; normalize to plain jsonl."""
    with open(path, "rb") as fh:
        magic = fh.read(4)
    if magic[:2] == b"PK":
        with zipfile.ZipFile(path) as zf:
            name = zf.namelist()[0]
            out = path.with_suffix(".jsonl")
            with zf.open(name) as src, open(out, "wb") as dst:
                shutil.copyfileobj(src, dst)
        path.unlink()
        return out
    if magic[:2] == b"\x1f\x8b":
        out = path.with_suffix(".jsonl")
        with gzip.open(path, "rb") as src, open(out, "wb") as dst:
            shutil.copyfileobj(src, dst)
        path.unlink()
        return out
    return path  # already plain jsonl


def ingest_series(series_id: str, api_key: str, entry: dict) -> None:
    """Download each game's riot events feed, process, delete raw."""
    resp = api_request("GET", GRID_FILE_LIST_URL.format(series_id=series_id), api_key)
    files = resp.json().get("files", [])
    games = entry.setdefault("games", {})
    event_files = [f for f in files
                   if "events" in f.get("id", "") and "riot" in f.get("id", "")]
    if not event_files:
        entry["status"] = "no_event_files"
        return

    for f in event_files:
        m = re.search(r"(\d+)$", f.get("id", ""))
        if not m:
            continue
        game_num = int(m.group(1))
        gkey = str(game_num)
        if games.get(gkey, {}).get("status") == "ok":
            continue
        if f.get("status") != "ready":
            games[gkey] = {"status": f"file_{f.get('status', 'unavailable')}"}
            continue

        with tempfile.TemporaryDirectory(prefix="grid_raw_") as tmp:
            raw = Path(tmp) / f"events_{series_id}_{game_num}_riot.dl"
            dl = api_request("GET", f["fullURL"], api_key, stream=True)
            with open(raw, "wb") as out:
                for chunk in dl.iter_content(chunk_size=1 << 20):
                    out.write(chunk)
            raw = decompress_if_needed(raw)
            try:
                r = process_game(raw, int(series_id), game_num, PROCESSED_DIR)
            except Exception as e:
                games[gkey] = {"status": f"process_error: {e}"}
                continue
            # raw feed is deleted with the TemporaryDirectory here — never retained
        if r is None:
            games[gkey] = {"status": "no_game_end"}
        else:
            games[gkey] = {"status": "ok", "T": r["T"], "label": r["label"]}

    ok = sum(1 for g in games.values() if g["status"] == "ok")
    entry["status"] = "done" if ok else "no_usable_games"


def stage_ingest(state: dict, limit: int | None) -> None:
    api_key = require_env("GRID_API_KEY")
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    pending = [sid for sid, e in state["series"].items()
               if e.get("eligible") and e.get("status") in ("pending", None)]
    if limit:
        pending = pending[:limit]
    print(f"ingesting {len(pending)} series")
    for i, sid in enumerate(pending, 1):
        try:
            ingest_series(sid, api_key, state["series"][sid])
        except Exception as e:
            state["series"][sid]["status"] = f"error: {e}"
        save_state(state)
        print(f"[{i}/{len(pending)}] series {sid}: {state['series'][sid]['status']}",
              flush=True)
    done = sum(1 for e in state["series"].values() if e.get("status") == "done")
    print(f"ingest done: {done} series with at least one processed game")


# ---------------------------------------------------------------- stage: pack

def stage_pack(state: dict, strict: bool) -> None:
    """Shard processed games + manifest (with patch + split) + features/norm."""
    game_dirs = sorted(d for d in PROCESSED_DIR.iterdir()
                       if d.is_dir() and (d / "sequence.npz").exists()
                       and (d / "meta.json").exists())
    if not game_dirs:
        sys.exit(f"no processed games in {PROCESSED_DIR}")

    verified = {sid for sid, e in state["series"].items() if e.get("eligible")}
    if strict and verified:
        before = len(game_dirs)
        game_dirs = [d for d in game_dirs if d.name.split("_")[0] in verified]
        print(f"--strict: kept {len(game_dirs)}/{before} games verified as "
              f"ESPORTS-in-window via fetch state")
    elif verified:
        unverified = [d.name for d in game_dirs if d.name.split("_")[0] not in verified]
        if unverified:
            print(f"WARNING: {len(unverified)} games not present in the fetched "
                  f"eligible-series list (pre-existing local data?). They will be "
                  f"packed anyway; use --strict to exclude. First few: {unverified[:5]}")

    if PACK_DIR.exists():
        shutil.rmtree(PACK_DIR)
    shards_dir = PACK_DIR / "shards"
    shards_dir.mkdir(parents=True)

    manifest_games = []
    for i in range(0, len(game_dirs), GAMES_PER_SHARD):
        shard_name = f"shard-{i // GAMES_PER_SHARD:04d}.tar"
        with tarfile.open(shards_dir / shard_name, "w") as tf:
            for d in game_dirs[i:i + GAMES_PER_SHARD]:
                tf.add(d, arcname=d.name)
                meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
                series_id = d.name.split("_")[0]
                manifest_games.append({
                    "game": d.name,
                    "series_id": series_id,
                    "game_num": meta.get("game_num"),
                    "shard": f"shards/{shard_name}",
                    "label": meta.get("label"),
                    "duration_s": meta.get("duration_s"),
                    "patch": meta.get("patch"),
                    "split": split_for(series_id),
                })
        print(f"  wrote {shard_name}")

    write_features_json(PACK_DIR)
    write_norm_json(PACK_DIR, game_dirs)

    counts = {s: sum(1 for g in manifest_games if g["split"] == s)
              for s in ("train", "val", "test")}
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "n_games": len(manifest_games),
        "split_counts": counts,
        "scope": "processed derivatives only; non-scrim (ESPORTS) series, "
                 f"past {os.environ.get('YEARS_BACK', '5')} years",
        "games": manifest_games,
    }
    with open(PACK_DIR / "manifest.json", "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=1)

    (PACK_DIR / "README.md").write_text(README_TEMPLATE.format(
        n=len(manifest_games), schema=SCHEMA_VERSION,
        years=os.environ.get("YEARS_BACK", "5"), **counts), encoding="utf-8")
    print(f"pack done: {len(manifest_games)} games, splits {counts} -> {PACK_DIR}")


README_TEMPLATE = """\
# LoL Win-Probability Training Data (processed)

Per-second game-state tensors for training win-likelihood models on professional
League of Legends games. {n} games, schema {schema}.
Splits (assigned by series id): {train} train / {val} val / {test} test.

Published as **processed derivatives only** (no raw event feeds), covering
**non-scrim (ESPORTS) series** from the past {years} years, with permission from the
data provider. Patch/version metadata is retained per game (`meta.json` and the
manifest's `patch` field).

## Layout
- `manifest.json` — one row per game: shard location, label, duration, patch, split
- `features.json` — ordered feature-name lists for every array
- `norm.json` — per-feature mean/std over the full corpus
- `shards/shard-NNNN.tar` — ~100 games each; members are `<seriesId>_<gameN>/`
  containing `meta.json` (draft, runes, lanes, patch, label) and `sequence.npz`
  (`X_champ [T,10,26]`, `X_champ_kda`, `X_items`, `X_item_cd`, `X_team [T,2,26]`,
  `X_global [T,6]`, `wards [W,6]` interval table, `y`)
"""


# ---------------------------------------------------------------- stage: upload

def stage_upload() -> None:
    token = require_env("HF_TOKEN")
    repo_id = require_env("HF_DATASET_REPO")
    private = os.environ.get("HF_PRIVATE", "0") == "1"
    if not (PACK_DIR / "manifest.json").exists():
        sys.exit("nothing packed — run the pack stage first")
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    api.create_repo(repo_id, repo_type="dataset", private=private, exist_ok=True)
    manifest = json.loads((PACK_DIR / "manifest.json").read_text(encoding="utf-8"))
    commit = api.upload_folder(
        folder_path=str(PACK_DIR),
        repo_id=repo_id,
        repo_type="dataset",
        commit_message=f"{manifest['n_games']} games, schema "
                       f"{manifest['schema_version']}, {manifest['created_utc']}",
    )
    print(f"upload done -> https://huggingface.co/datasets/{repo_id}")
    print(f"pin this revision in training: {getattr(commit, 'oid', commit)}")


# ---------------------------------------------------------------- main

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["fetch", "ingest", "pack", "upload", "all"])
    ap.add_argument("--limit", type=int, help="ingest at most N series (testing)")
    ap.add_argument("--strict", action="store_true",
                    help="pack only games verified eligible by the fetch stage")
    args = ap.parse_args()

    state = load_state()
    if args.stage in ("fetch", "all"):
        stage_fetch(state)
    if args.stage in ("ingest", "all"):
        stage_ingest(state, args.limit)
    if args.stage in ("pack", "all"):
        stage_pack(state, args.strict)
    if args.stage in ("upload", "all"):
        stage_upload()


if __name__ == "__main__":
    main()
