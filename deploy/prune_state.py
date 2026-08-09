"""Prune pipeline_state.json to series starting on/after an absolute cutoff date.

The pipeline's window is set by YEARS_BACK (whole years). This applies an exact
date floor on top, so the ingest scope is reproducible from code alone — no need
to ship the state file. Reproduce the current 2026-only scope on a fresh VM with:

    YEARS_BACK=1 python -m pipeline.pipeline fetch      # lists ~2025-08 .. now
    python deploy/prune_state.py --after 2026-01-01     # keep 2026 only

Removes series whose startTimeScheduled is before --after. Idempotent.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STATE = ROOT / "data" / "pipeline_state.json"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--after", required=True,
                    help="keep series with start_time >= this date, e.g. 2026-01-01")
    ap.add_argument("--state", default=str(STATE), help="path to pipeline_state.json")
    args = ap.parse_args()

    # normalize a bare date to a full ISO-8601 UTC instant for lexicographic compare
    cutoff = args.after
    if len(cutoff) == 10:  # YYYY-MM-DD
        cutoff += "T00:00:00Z"

    path = Path(args.state)
    state = json.loads(path.read_text(encoding="utf-8"))
    series = state["series"]
    before = len(series)

    kept = {sid: e for sid, e in series.items()
            if e.get("start_time") and e["start_time"] >= cutoff}
    state["series"] = kept

    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=1), encoding="utf-8")
    tmp.replace(path)

    times = sorted(e["start_time"] for e in kept.values())
    print(f"cutoff:   {cutoff}")
    print(f"before:   {before}")
    print(f"removed:  {before - len(kept)}")
    print(f"kept:     {len(kept)}")
    if times:
        print(f"earliest: {times[0]}  |  latest: {times[-1]}")


if __name__ == "__main__":
    main()
