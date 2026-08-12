"""File-based prioritized job queue for the GPU window.

Feeds the card from a YAML backlog so it never idles; supports N concurrent
capped processes (validated on Day 1, sequential fallback). Every job's
stdout/stderr goes to its own log; a ledger records outcomes.

jobs.yaml format:
  jobs:
    - name: ssl_M_seed1
      cmd: python -m pipeline.winprob.train_ssl --trunk M --run runs/ssl_M_s1 ...
      priority: 10          # higher runs first
      deps: [pack]          # names that must have succeeded first
      env: {HSA_VISIBLE_DEVICES: "0"}   # optional per-job env

Usage:
  python -m pipeline.winprob.runner --jobs configs/winprob/jobs_day1.yaml \
      --concurrent 2 --ledger runs/ledger.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml


def now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def load_ledger(path: Path) -> dict[str, str]:
    done: dict[str, str] = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                done[row["name"]] = row["status"]
    return done


def append_ledger(path: Path, **row):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps({"utc": now(), **row}) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--jobs", required=True)
    ap.add_argument("--concurrent", type=int, default=1)
    ap.add_argument("--ledger", default="runs/ledger.jsonl")
    ap.add_argument("--logs", default="runs/logs")
    ap.add_argument("--poll", type=float, default=10.0)
    args = ap.parse_args()

    with open(args.jobs, encoding="utf-8") as fh:
        jobs = yaml.safe_load(fh)["jobs"]
    names = [j["name"] for j in jobs]
    if len(set(names)) != len(names):
        raise SystemExit("duplicate job names")
    ledger_path, logs_dir = Path(args.ledger), Path(args.logs)
    logs_dir.mkdir(parents=True, exist_ok=True)
    status = load_ledger(ledger_path)          # resumable: succeeded jobs skip

    running: dict[str, tuple[subprocess.Popen, object]] = {}
    pending = [j for j in jobs if status.get(j["name"]) != "succeeded"]
    pending.sort(key=lambda j: -j.get("priority", 0))
    print(f"{len(pending)} pending, {len(jobs) - len(pending)} already done")

    while pending or running:
        # reap
        for name in list(running):
            proc, logfh = running[name]
            rc = proc.poll()
            if rc is None:
                continue
            logfh.close()
            st = "succeeded" if rc == 0 else "failed"
            status[name] = st
            append_ledger(ledger_path, name=name, status=st, rc=rc)
            print(f"[{now()}] {name}: {st} (rc={rc})")
            del running[name]

        # launch
        while len(running) < args.concurrent and pending:
            ready = next((j for j in pending
                          if all(status.get(d) == "succeeded" for d in j.get("deps", []))
                          and not any(d in running for d in j.get("deps", []))), None)
            if ready is None:
                blocked = [j["name"] for j in pending
                           if any(status.get(d) == "failed" for d in j.get("deps", []))]
                for name in blocked:
                    pending[:] = [j for j in pending if j["name"] != name]
                    status[name] = "blocked"
                    append_ledger(ledger_path, name=name, status="blocked")
                    print(f"[{now()}] {name}: blocked (failed dependency)")
                break
            pending.remove(ready)
            env = {**os.environ, **{k: str(v) for k, v in (ready.get("env") or {}).items()}}
            logfh = open(logs_dir / f"{ready['name']}.log", "a", encoding="utf-8")
            logfh.write(f"\n===== {now()} launch: {ready['cmd']}\n")
            logfh.flush()
            proc = subprocess.Popen(ready["cmd"], shell=True, env=env,
                                    stdout=logfh, stderr=subprocess.STDOUT)
            running[ready["name"]] = (proc, logfh)
            append_ledger(ledger_path, name=ready["name"], status="started", pid=proc.pid)
            print(f"[{now()}] {ready['name']}: started (pid {proc.pid})")

        if running:
            time.sleep(args.poll)
        elif pending:
            # nothing ready and nothing running -> everything left is blocked
            for j in pending:
                append_ledger(ledger_path, name=j["name"], status="blocked")
                print(f"{j['name']}: blocked (unmet dependency)")
            break

    n_ok = sum(1 for s in status.values() if s == "succeeded")
    print(f"queue drained: {n_ok}/{len(jobs)} succeeded")


if __name__ == "__main__":
    main()
