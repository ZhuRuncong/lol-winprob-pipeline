"""Pre-flight GRID bandwidth check for the cloud VM.

Downloads one real event feed and measures throughput. This is the gate that
decides whether moving to a cloud VM actually helps: locally the link caps at
~0.6 MB/s, and we could NOT measure GRID's server-side ceiling from there. Run
this FIRST on the VM. If it reports fast throughput, GRID serves faster than a
home link and the full ingest is worth launching; if it still reports ~0.6 MB/s,
GRID itself is the cap and a VM will not help.

Exit code 0 = fast enough (>= THRESHOLD), 1 = too slow (abort the run).
"""
from __future__ import annotations
import os, sys, time, json
from pathlib import Path

THRESHOLD_MBPS = 5.0  # require >= 5 MB/s to justify the cloud run (local was ~0.6)
ROOT = Path(__file__).resolve().parent.parent


def load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def main() -> int:
    load_dotenv(ROOT / ".env")
    key = os.environ.get("GRID_API_KEY")
    if not key:
        print("GRID_API_KEY not set (export it or put it in .env)")
        return 1
    import requests

    state_path = ROOT / "data" / "pipeline_state.json"
    if not state_path.exists():
        print(f"no {state_path} — copy your pruned 2026 state to the VM first "
              f"(or run: python -m pipeline.pipeline fetch)")
        return 1
    state = json.loads(state_path.read_text(encoding="utf-8"))
    sid = next((s for s, e in state["series"].items()
                if e.get("status") in ("pending", None)), None)
    if sid is None:
        print("no pending series in state — nothing to validate")
        return 1

    hdr = {"x-api-key": key}
    r = requests.get(f"https://api.grid.gg/file-download/list/{sid}",
                     headers=hdr, timeout=120)
    r.raise_for_status()
    ev = [f for f in r.json().get("files", [])
          if "events" in f.get("id", "") and "riot" in f.get("id", "")]
    if not ev:
        print(f"series {sid} has no riot event feed; try another")
        return 1

    print(f"measuring GRID download speed on series {sid} ...")
    t0 = time.monotonic()
    nbytes = 0
    dl = requests.get(ev[0]["fullURL"], headers=hdr, stream=True, timeout=120)
    for chunk in dl.iter_content(chunk_size=1 << 20):
        nbytes += len(chunk)
        if time.monotonic() - t0 >= 20:  # 20s sample is enough
            break
    dl.close()
    dt = time.monotonic() - t0
    mbps = (nbytes / 1e6) / dt
    print(f"  {nbytes/1e6:.1f} MB in {dt:.1f}s = {mbps:.2f} MB/s ({mbps*8:.1f} Mbit/s)")
    print(f"  local reference was ~0.60 MB/s; threshold to proceed is {THRESHOLD_MBPS} MB/s")

    if mbps >= THRESHOLD_MBPS:
        est_gb = 470
        hrs = est_gb * 1000 / mbps / 3600
        print(f"\nFAST ENOUGH. Full 2026 ingest (~{est_gb} GB) ~= {hrs:.1f} h of transfer.")
        print("Proceed: launch the ingest (see RUNBOOK.md).")
        return 0
    print("\nTOO SLOW — GRID itself appears to cap near this rate, or this VM's "
          "egress is also limited. A cloud VM will NOT meaningfully help; "
          "reconsider scope instead of running the full ingest.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
