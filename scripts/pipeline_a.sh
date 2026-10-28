#!/usr/bin/env bash
# Unattended chain A (run on one GPU from the project folder):
#   1) ET threshold tuning on fold-0 validation predictions
#   2) when each baseline finishes: test inference + scoring, then delete its last.pt (frees quota)
#   3) paired statistics of XLite (fold 0) against all baselines
# Start:  bash scripts/run.sh 0 pipe_a bash scripts/pipeline_a.sh      Progress: cat logs/pipe_a.log
ZIP=${BRATS_ZIP:-$HOME/data/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData.zip}
mkdir -p results
wait_done()  { until grep -q "^done:" "logs/$1.log" 2>/dev/null; do sleep 300; done; }
wait_count() { until [ "$(ls "$1" 2>/dev/null | wc -l)" -ge "$2" ]; do sleep 120; done; }
say() { echo "[$(date '+%d %b %H:%M')] $*"; }

say "waiting for validation predictions"
wait_count preds/xlite_f0_val 202
wait_count preds/unet3d_f0_val 202
python tools/tune_et.py --pred_dir preds/xlite_f0_val  --data_root $ZIP --fold 0 2>&1 | grep -v -i warn > results/tune_et_xlite.txt
python tools/tune_et.py --pred_dir preds/unet3d_f0_val --data_root $ZIP --fold 0 2>&1 | grep -v -i warn > results/tune_et_unet3d.txt
say "ET tuning written to results/tune_et_*.txt"

for pair in segresnet:segresnet_f0 unetr:unetr_f0 swinunetr:swin_f0; do
  run=${pair%%:*}; job=${pair##*:}
  say "waiting for $job"
  wait_done "$job"
  python infer.py --checkpoint runs/$run/fold0/best.pt --data_root $ZIP --which test \
      --out_dir preds/${run}_test --tta 8 --no_unc > logs/inf_${run}.log 2>&1
  python evaluate.py --pred_dir preds/${run}_test --data_root $ZIP --which test \
      --out results/${run}_test.csv 2>&1 | grep -v -i warn
  rm -f runs/$run/fold0/last.pt
  say "$run scored -> results/${run}_test.csv"
done

python stats.py --proposed results/xlite_f0_test.csv --baselines results/unet3d_test.csv \
    results/segresnet_test.csv results/unetr_test.csv results/swinunetr_test.csv \
    --out results/stats_f0_all.csv
rm -f runs/unet3d/fold0/last.pt
say "ALL DONE (chain A)"
