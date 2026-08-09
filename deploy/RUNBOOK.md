# AWS EC2 ingest runbook

Why: the `ingest` stage is bottlenecked by local internet (~0.6 MB/s). Each riot
event feed is ~96 MB and the 2026 scope is ~470 GB, so on a home link it's ~9
days. An EC2 instance has datacenter egress (typically 50–200× a home link).
GRID's server-side ceiling is unknown from a slow link, so **step 4 validates it
before you commit to a long run.**

Raw feeds are streamed to a temp dir and deleted per game, so the instance does
**not** need 470 GB of disk — only the processed output (~4 GB) plus pack shards
(~4 GB). Compute is trivial (process_game ≈ 2.4s/game); this is entirely a
bandwidth play.

## 0. Launch an EC2 instance
- **AMI:** Ubuntu Server 24.04 LTS (x86_64)
- **Type:** `t3.small` (2 vCPU / 2 GB) is plenty; `t3.medium` if you want headroom
- **Storage:** 40 GB gp3 root volume
- **Security group:** inbound SSH (port 22) from **your IP only**
- **Key pair:** create/download one, then `chmod 400 your-key.pem`

SSH in:
```bash
ssh -i your-key.pem ubuntu@<EC2_PUBLIC_IP>
```

## 1. Clone the code
```bash
sudo apt-get update -y && sudo apt-get install -y git
git clone <YOUR_REPO_URL> lol-winprob-pipeline
cd lol-winprob-pipeline
```
The repo is **code only** — no secrets, no GRID data. You bring the keys (step 2)
and regenerate the series list from the API (step 3).

## 2. Secrets (env vars — never committed)
```bash
export GRID_API_KEY='...'                      # your GRID key
export HF_TOKEN='...'                           # your HF write token
export HF_DATASET_REPO='zauberine/lol-winprob'
```
Put these in `~/.bashrc` if you want them to survive re-login.

## 3. Install deps + regenerate the 2026 series list
```bash
sudo apt-get install -y python3-venv tmux
python3 -m venv .venv && source .venv/bin/activate
pip install -U pip && pip install -r requirements.txt

# list eligible ESPORTS series, then floor to this-year-only (reproduces the
# 3,265-series 2026 scope). YEARS_BACK=1 keeps the fetch short (~2025-08 .. now).
YEARS_BACK=1 python -m pipeline.pipeline fetch
python deploy/prune_state.py --after 2026-01-01
```

## 4. Speed gate (decision point)
```bash
python deploy/check_speed.py
```
Downloads one feed and prints MB/s:
- **≥ 5 MB/s** → GRID serves fast; proceed. It prints the estimated total hours.
- **~0.6 MB/s** → GRID itself is the cap; EC2 won't help. **Stop** and reconsider
  scope instead of burning instance hours.

(Steps 3–4 are also wrapped in `bash deploy/bootstrap.sh` if you prefer one call.)

## 5. Run the ingest (durable)
Launch inside `tmux` so it survives SSH disconnects; resumable via
`pipeline_state.json` if interrupted:
```bash
tmux new -s ingest
source .venv/bin/activate
python -m pipeline.pipeline ingest
#   detach: Ctrl-b then d   |   reattach: tmux attach -t ingest
```
Progress prints per series; state saves after each, so Ctrl-c + rerun continues.

## 6. Pack + upload (from the instance)
```bash
python -m pipeline.pipeline pack --strict     # 2026-eligible games only
python -m pipeline.pipeline upload            # pushes to zauberine/lol-winprob
```
`upload` prints the dataset URL and a commit revision — **pin that revision** in
training. The processed data on the instance is disposable; the published HF
dataset is the deliverable.

## 7. Terminate the instance
Stop/terminate the EC2 instance when the upload finishes so it stops billing.
Nothing else needs to persist.

---
### Notes
- **Cost:** a `t3.small` is ~$0.02/hr; even a full multi-hour ingest is a couple
  of dollars. The volume + data transfer *in* are negligible; you pay mostly for
  instance-hours. Terminate when done.
- **Scope guard:** `pack --strict` keeps only games whose series is in the fetched
  2026 eligible set — enforces the non-scrim / in-window publish permission.
- **Rate limit:** the pipeline self-throttles to 20 req/min (50% of GRID's 40/60s
  cap); safe to run unattended.
- **Retries:** series that fail mid-run get an error status and are skipped;
  rerun `ingest` later to retry only those.
