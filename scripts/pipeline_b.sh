#!/usr/bin/env bash
# Unattended chain B (run on one GPU from the project folder):
#   1) when the kernel-3 ablation finishes: test inference + scoring
#   2) when fold 4 finishes: five-fold XLite ensemble on the test set + scoring
# Start:  bash scripts/run.sh 1 pipe_b bash scripts/pipeline_b.sh      Progress: cat logs/pipe_b.log
ZIP=${BRATS_ZIP:-$HOME/data/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData.zip}
mkdir -p results
wait_done() { until grep -q "^done:" "logs/$1.log" 2>/dev/null; do sleep 300; done; }
say() { echo "[$(date '+%d %b %H:%M')] $*"; }

say "waiting for xlite_k3_f0"
wait_done xlite_k3_f0
python infer.py --checkpoint runs/xlite_k3/fold0/best.pt --data_root $ZIP --which test \
    --out_dir preds/xlite_k3_test --tta 8 --no_unc > logs/inf_xlite_k3.log 2>&1
python evaluate.py --pred_dir preds/xlite_k3_test --data_root $ZIP --which test \
    --out results/xlite_k3_test.csv 2>&1 | grep -v -i warn
say "kernel-3 ablation scored -> results/xlite_k3_test.csv"

say "waiting for xlite_f4"
wait_done xlite_f4
python infer.py --checkpoint runs/xlite/fold{0,1,2,3,4}/best.pt --data_root $ZIP --which test \
    --out_dir preds/xlite_ens_test --tta 8 --no_unc > logs/inf_xlite_ens.log 2>&1
python evaluate.py --pred_dir preds/xlite_ens_test --data_root $ZIP --which test \
    --out results/xlite_ens_test.csv 2>&1 | grep -v -i warn
rm -f runs/xlite/fold*/last.pt runs/xlite_k3/fold0/last.pt
say "ALL DONE (chain B)"
