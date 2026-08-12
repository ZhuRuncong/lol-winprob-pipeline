#!/usr/bin/env bash
set -euo pipefail
cd /app
OUT="${OUT_DIR:-/app/out}"
mkdir -p "$OUT"
exec > >(tee -a "$OUT/results.txt") 2>&1

TRUNK="${TRUNK:-M}"
SSL_EPOCHS="${SSL_EPOCHS:-60}"
K="${K_MEMBERS:-8}"
B=data/winprob_bundles
R=results

echo "=== winprob pipeline start $(date -u +%FT%TZ)  trunk=$TRUNK ssl_epochs=$SSL_EPOCHS members=$K ==="
python -c "import torch; print('torch', torch.__version__, 'gpu', torch.cuda.is_available())"

echo "--- [1/7] baselines ---"
python -m pipeline.winprob.baselines.time_prior --bundles $B --out $R
python -m pipeline.winprob.baselines.logistic   --bundles $B --out $R
python -m pipeline.winprob.baselines.gbt        --bundles $B --out $R
python -m pipeline.winprob.eval.report --bundles $B --preds $R/gbt --split test

echo "--- [2/7] phase A: SSL pretrain ---"
python -m pipeline.winprob.train_ssl --config configs/winprob/ssl_M.yaml \
    --trunk "$TRUNK" --epochs "$SSL_EPOCHS" --bundles $B --run runs/ssl --resume

echo "--- [3/7] phase B: $K finetune members ---"
MEMBER_CKPTS=()
MEMBER_PREDS=()
for i in $(seq 1 "$K"); do
    python -m pipeline.winprob.train_finetune --init runs/ssl/best.pt \
        --config configs/winprob/finetune.yaml --bundles $B \
        --run "runs/ft_s$i" --seed "$i" --preds-out "$R/member_s$i"
    MEMBER_CKPTS+=("runs/ft_s$i/best.pt")
    MEMBER_PREDS+=("$R/member_s$i")
done

echo "--- [4/7] ensemble + calibration ---"
python -m pipeline.winprob.ensemble --members "${MEMBER_PREDS[@]}" --out $R/ensemble
python -m pipeline.winprob.calibrate --preds $R/ensemble --bundles $B --out $R/ensemble_cal

echo "--- [5/7] final report (decision rule vs GBT) ---"
python -m pipeline.winprob.eval.report --bundles $B --preds $R/ensemble_cal \
    --compare $R/gbt --split test

echo "--- [6/7] latent-state probes ---"
python -m pipeline.winprob.probes horizon --ckpt runs/ssl/best.pt --bundles $B --out runs/probes
python -m pipeline.winprob.probes fog     --ckpt runs/ssl/best.pt --bundles $B --out runs/probes
python -m pipeline.winprob.probes event-gain --preds $R/ensemble_cal \
    --ref-preds $R/gbt --bundles $B --out runs/probes

echo "--- [7/7] export model ---"
python -m pipeline.winprob.export_model --members "${MEMBER_CKPTS[@]}" \
    --bundles $B --calibration $R/ensemble_cal/calibration_report.json \
    --out "$OUT/model.pt"
cp $R/ensemble_cal/report.json "$OUT/final_report.json"
cp $R/ensemble_cal/calibration_report.json "$OUT/calibration_report.json"

echo "=== DONE $(date -u +%FT%TZ): $OUT/results.txt  $OUT/model.pt ==="
