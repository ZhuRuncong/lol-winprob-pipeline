#!/usr/bin/env bash
# Container entrypoint for Fargate. Regenerates the 2026 series list if absent,
# then runs the requested stage(s). Secrets arrive as env vars from SSM.
#
# Usage (as the task command / containerOverrides):
#   gate    - just the GRID speed gate (~1 min, cheap pre-flight)
#   all     - gate -> ingest -> pack --strict -> upload   (default)
#   ingest | pack | upload | fetch - a single pipeline stage
set -euo pipefail
cd /app

: "${GRID_API_KEY:?GRID_API_KEY not set (inject via SSM)}"
# START_AFTER is the single source of truth for scope; fetch floors to it.
export START_AFTER="${START_AFTER:-2026-01-01}"

# On Fargate the filesystem is ephemeral (unless /app/data is an EFS mount), so
# regenerate the series list on first run.
if [ ! -f data/pipeline_state.json ]; then
  echo "== regenerating series list (start time on/after ${START_AFTER}) =="
  python -m pipeline.pipeline fetch
fi

STAGE="${1:-all}"
echo "== stage: ${STAGE} =="
case "${STAGE}" in
  gate)
    exec python deploy/check_speed.py
    ;;
  all)
    python deploy/check_speed.py          # fail-fast: aborts task if GRID is slow
    python -m pipeline.pipeline ingest
    python -m pipeline.pipeline pack --strict
    python -m pipeline.pipeline upload
    ;;
  *)
    exec python -m pipeline.pipeline "${STAGE}"
    ;;
esac
