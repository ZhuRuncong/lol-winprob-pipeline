# lol-winprob-pipeline

Self-contained pipeline that turns GRID League of Legends event feeds into per-second
game-state tensors and publishes them as a Hugging Face dataset for training a
win-likelihood transformer.

**Publishes processed derivatives only** (`meta.json` + `sequence.npz`) — raw event
feeds are downloaded to a temp dir, processed, and deleted immediately. Scope is
restricted to **non-scrim (ESPORTS) series from the past 5 years**, and patch/version
metadata is retained per game.

## Layout

```
pipeline/
  build_processed.py   # raw events jsonl -> meta.json + sequence.npz (streaming, 1 pass)
  pipeline.py          # fetch -> ingest -> pack -> upload
data/
  processed/           # <seriesId>_<gameN>/{meta.json, sequence.npz}
  pack/                # shards + manifest.json + features.json + norm.json (upload root)
  pipeline_state.json  # resumable per-series ingest state
.env                   # your keys (gitignored) — copy from .env.example
```

## Keys

Copy `.env.example` to `.env` and fill in. Loaded automatically from the project root;
real process env vars override the file.

| Key | Needed for | Notes |
|-----|-----------|-------|
| `GRID_API_KEY` | fetch, ingest | GRID API key |
| `HF_TOKEN` | upload | HF token with **write** scope |
| `HF_DATASET_REPO` | upload | `user_or_org/name` |
| `HF_PRIVATE` | upload (opt) | `1` = private repo, default `0` |
| `GRID_TITLE_ID` | opt | default `3` (League of Legends) |
| `YEARS_BACK` | opt | default `5` |
| `WORK_DIR` | opt | defaults to `./data` |

## Usage

```bash
pip install -r requirements.txt

python -m pipeline.pipeline fetch              # list eligible ESPORTS series (past 5y)
python -m pipeline.pipeline ingest --limit 2   # smoke-test the API contract
python -m pipeline.pipeline ingest             # download -> process -> delete raw
python -m pipeline.pipeline pack --strict      # shard + manifest + norm, eligible-only
python -m pipeline.pipeline upload             # push data/pack/ to the HF dataset repo
# or: python -m pipeline.pipeline all
```

Every stage is idempotent and resumable via `data/pipeline_state.json`.

## Reprocess local games without the API

`build_processed.py` also runs standalone against local raw feeds:

```bash
python -m pipeline.build_processed --one <seriesId>_<gameN>
python -m pipeline.build_processed --all      # over data/raw/<seriesId>/events_*_riot.jsonl
```

## Training container

Consumers pull the published artifact only — never the game API:

```python
from huggingface_hub import snapshot_download
path = snapshot_download("youruser/lol-winprob", repo_type="dataset", revision="<pin>")
# read manifest.json -> open shards -> normalize with norm.json -> rasterize wards
```
