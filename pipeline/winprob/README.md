# winprob — win-probability model training stack

Causal p(blue wins | state up to t) at every second, trained on the published
corpus (`zauberine/lol-winprob`, pinned revision in `config.py`).

Design doctrine (see the approved plan): **games, not timesteps, are the unit
of supervision.** ~12M seconds share only 6,242 outcome labels, so the trunk
is fit by dense self-supervision (phase A) and the win label only trains a
small head + low-LR finetune (phase B). No label smoothing anywhere; the
deliverable is a calibrated probability. The transformer ships only if it
beats the GBDT baseline with a series-bootstrap 95% CI excluding zero in the
10–25 minute buckets.

## Local smoke (Day 0, this repo's 142-game sample)

```bash
python -m pipeline.winprob.data.stats --local-sample --out data/winprob_bundles_local/norm_v2.json
python -m pipeline.winprob.data.pack  --local-sample --out data/winprob_bundles_local
python -m pytest tests/test_winprob_contracts.py -x -q
python -m pipeline.winprob.baselines.gbt --bundles data/winprob_bundles_local --out data/winprob_results_local
python -m pipeline.winprob.train_ssl --config configs/winprob/smoke.yaml --bundles data/winprob_bundles_local --run runs/smoke_ssl
python -m pipeline.winprob.train_finetune --init runs/smoke_ssl/best.pt --bundles data/winprob_bundles_local --run runs/smoke_ft --preds-out data/winprob_results_local/member_smoke
python -m pipeline.winprob.ensemble --members data/winprob_results_local/member_smoke --out data/winprob_results_local/ensemble
python -m pipeline.winprob.calibrate --preds data/winprob_results_local/ensemble --bundles data/winprob_bundles_local --out data/winprob_results_local/ensemble_cal
python -m pipeline.winprob.eval.report --bundles data/winprob_bundles_local --preds data/winprob_results_local/ensemble_cal --compare data/winprob_results_local/gbt
```

No number from the 142-game sample is reported — it proves plumbing.

## The 4-day GPU window (ROCm box)

Build + enter the container (`docker/Dockerfile.rocm`), then:

- **Day 1** — burn-in (`python -c "import torch; ..."`, tests in-container,
  throughput per trunk size, 2-process concurrency trial), then
  `python -m pipeline.winprob.runner --jobs configs/winprob/jobs_day1.yaml --concurrent 2`.
  Gate: pick S/M/L by frozen-probe val log-loss + SSL val curves (`runs/*/metrics.jsonl`).
- **Day 2** — full pretrain of the chosen config, seeds 1+2; horizon-window
  ({8,32,128,full} via `model.attn_window`) and `context_one` variants;
  phase-B recipe sweep on an intermediate checkpoint. Gate: freeze the
  finetune recipe by val 10–25 min log-loss.
- **Day 3** — K=8 `train_finetune` members (6 from trunk seed 1, 2 from seed
  2) → `ensemble` → the ablation grid: no-SSL from-scratch
  (`train_finetune` without `--init`, same trunk preset), context-1,
  corpus-fraction curves.
- **Day 4** — temporal split (splits.temporal_split), `calibrate`, `probes`
  (horizon / fog / event-gain), final `eval.report --compare` vs the GBDT with
  the pre-registered CI decision rule. Reserve the last 4–6 h as buffer.

Every run directory carries `metrics.jsonl`; every checkpoint embeds config,
norm_v2 version, and git SHA. The runner's `runs/ledger.jsonl` is the record
of what ran.
