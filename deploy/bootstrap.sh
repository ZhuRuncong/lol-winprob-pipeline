#!/usr/bin/env bash
# Setup + state regeneration + speed gate on a fresh EC2 (Ubuntu) instance.
# Prereq: GRID_API_KEY / HF_TOKEN / HF_DATASET_REPO exported (see RUNBOOK step 2).
# Run from the project root:  bash deploy/bootstrap.sh
set -euo pipefail

echo "== 1. system deps =="
sudo apt-get update -y
sudo apt-get install -y python3 python3-venv python3-pip tmux

echo "== 2. python venv + requirements =="
python3 -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
pip install --quiet --upgrade pip
pip install --quiet -r requirements.txt
echo "   $(python -c 'import huggingface_hub,numpy,requests; print("hf",huggingface_hub.__version__,"| numpy",numpy.__version__)')"

echo "== 3. secrets check =="
if [ -z "${GRID_API_KEY:-}" ] && [ ! -f .env ]; then
  echo "   ERROR: GRID_API_KEY unset and no .env. Export your keys first (RUNBOOK step 2)."
  exit 1
fi

echo "== 4. regenerate series list =="
# START_AFTER is the single source of truth for scope; fetch floors to it, so no
# separate prune step is needed. Defaults to 2026-01-01 (this year) if unset.
export START_AFTER="${START_AFTER:-2026-01-01}"
if [ -f data/pipeline_state.json ]; then
  echo "   data/pipeline_state.json already exists — skipping fetch (delete it to refetch)."
else
  echo "   fetching series with start time on/after ${START_AFTER}"
  python -m pipeline.pipeline fetch
fi

echo "== 5. GRID speed gate =="
python deploy/check_speed.py
echo
echo "Gate passed. Launch the ingest under tmux:"
echo "  tmux new -s ingest"
echo "  source .venv/bin/activate && python -m pipeline.pipeline ingest"
