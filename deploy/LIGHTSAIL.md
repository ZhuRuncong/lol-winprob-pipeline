# Running the ingest on AWS Lightsail

Lightsail is a simplified VM — the best fit of the three options for this job:

- **Persistent SSD**, so the pipeline's resumable state (`pipeline_state.json`)
  survives reboots with no EFS/S3 wiring (unlike Fargate).
- **Bundled data transfer**, and **inbound transfer doesn't count** against it —
  and this workload is ~470 GB *in*, ~8 GB *out*. So the big download is free and
  there are no metered-egress surprises.
- **Browser-based SSH** — no key-pair management needed.
- Flat, predictable price, billed hourly, so a few-hour run costs cents.

It reuses `deploy/bootstrap.sh` exactly as the EC2 flow does. See
[RUNBOOK.md](RUNBOOK.md) for the bandwidth background and
[fargate/](fargate/) for the container route.

---

## 1. Create the instance (console)
Console → **Lightsail** → **Create instance**.
- **Region:** any near you.
- **Platform:** Linux/Unix · **Blueprint:** **OS Only → Ubuntu 24.04 LTS**
  (not an app blueprint).
- **Plan:** pick one with **≥ 2 GB RAM and ≥ 40 GB SSD** (the ~$10–12/mo dual-stack
  tier is plenty). Compute is trivial here; you're paying mostly for a few hours
  of uptime and good network. Billed hourly, so a short run is cents.
- Name it `lol-winprob` → **Create instance**.

No firewall changes needed: SSH (22) is open by default and outbound is
unrestricted — that's all this job uses.

## 2. Connect
On the instance card → **Connect using SSH** (opens a browser terminal). Or use
your own client with the account's default key from
**Account → SSH keys**.

## 3. Clone the code
```bash
sudo apt-get update -y && sudo apt-get install -y git
git clone https://github.com/ZhuRuncong/lol-winprob-pipeline.git
cd lol-winprob-pipeline
```

## 4. Secrets (env vars — never committed)
```bash
export GRID_API_KEY='...'                      # your GRID key
export HF_TOKEN='...'                           # your HF write token
export HF_DATASET_REPO='zauberine/lol-winprob'
```
Append these to `~/.bashrc` if you want them to persist across reconnects.

## 5. Bootstrap: deps + 2026 series list + speed gate
```bash
bash deploy/bootstrap.sh
```
This installs Python + tmux, regenerates the 2026 series list from the API
(`fetch` + `prune_state.py --after 2026-01-01`), then runs the **speed gate**:
- **≥ 5 MB/s** → proceed.
- **~0.6 MB/s** → GRID itself is the cap; stop and reconsider scope (a VM won't
  help). Lightsail network is normally fast, so this should clear easily.

## 6. Run the ingest (durable)
Run inside `tmux` so it survives a dropped SSH/browser session; it's resumable
via `pipeline_state.json` on the persistent disk if anything interrupts it:
```bash
tmux new -s ingest
source .venv/bin/activate
python -m pipeline.pipeline ingest
#   detach: Ctrl-b then d   |   reattach: tmux attach -t ingest
```

## 7. Pack + upload
```bash
python -m pipeline.pipeline pack --strict     # 2026-eligible games only
python -m pipeline.pipeline upload            # pushes to zauberine/lol-winprob
```
`upload` prints the dataset URL and a commit revision — **pin that revision** in
training.

## 8. Delete the instance
Lightsail console → instance → **Delete**. Billing stops when it's deleted.
(If you might rerun soon, you can **Stop** instead — a stopped instance keeps its
disk and the resumable state, but note stopped instances still incur a small
storage charge.)

---
### Notes
- **Resumability is free here:** the disk persists, so Ctrl-c + rerun `ingest`, or
  even a reboot, continues from the saved state. This is why Lightsail beats
  Fargate for this bandwidth-bound, long-running job.
- **Snapshots (optional):** take a Lightsail snapshot after a long ingest if you
  want a restore point before pack/upload.
- **Scope guard:** `pack --strict` keeps only games whose series is in the fetched
  2026 eligible set — enforces the non-scrim / in-window publish permission.
