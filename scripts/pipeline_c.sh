#!/usr/bin/env bash
# Unattended chain C: uncertainty, explainability, efficiency, CV summary, figures.
# Start:  bash scripts/run.sh 2 pipe_c bash scripts/pipeline_c.sh      Progress: cat logs/pipe_c.log
# Everything ends up in results/ ; total about 8-10 hours.
ZIP=${BRATS_ZIP:-$HOME/data/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData.zip}
mkdir -p results
wait_done() { until grep -q "^done:" "logs/$1.log" 2>/dev/null; do sleep 300; done; }
say() { echo "[$(date '+%d %b %H:%M')] $*"; }
quiet() { grep -v -i -E "warn|TRT|CUDA 13"; }

say "1/6 uncertainty inference: XLite fold 0, 4 flips x 5 MC-dropout passes, test set"
python infer.py --checkpoint runs/xlite/fold0/best.pt --data_root $ZIP --which test \
    --out_dir preds/xlite_f0_unc --tta 4 --mc 5 --save_probs > logs/inf_xlite_unc.log 2>&1
python evaluate.py --pred_dir preds/xlite_f0_unc --data_root $ZIP --which test \
    --out results/xlite_f0_unc_test.csv 2>&1 | quiet

say "2/6 calibration and QU-BraTS score"
python eval_uncertainty.py --pred_dir preds/xlite_f0_unc --data_root $ZIP --which test \
    --dice_csv results/xlite_f0_unc_test.csv --out_dir results/unc_xlite 2>&1 | quiet
cat results/unc_xlite/uncertainty_summary.json

say "3/6 explainability: Seg-Grad-CAM, occlusion, Integrated Gradients, deletion, sanity check (50 cases)"
python explain.py --checkpoint runs/xlite/fold0/best.pt --data_root $ZIP --which test \
    --n_cases 50 --ig --sanity_cases 10 --figures 6 --unc_dir preds/xlite_f0_unc \
    --out_dir results/xai_xlite 2>&1 | quiet | grep -v "^\["

say "4/6 efficiency profile (params, GFLOPs, memory, GPU and 8-thread CPU time per volume)"
python profile_models.py --configs configs/xlite.yaml configs/xlite_k3.yaml configs/unet3d.yaml \
    configs/segresnet.yaml configs/unetr.yaml configs/swinunetr.yaml \
    --out results/efficiency.csv 2>&1 | quiet

say "5/6 figure: Dice vs parameters"
python make_figures.py --efficiency results/efficiency.csv --summaries \
    xlite=results/xlite_f0_test_summary.json xlite_k3=results/xlite_k3_test_summary.json \
    unet3d=results/unet3d_test_summary.json segresnet=results/segresnet_test_summary.json \
    unetr=results/unetr_test_summary.json swinunetr=results/swinunetr_test_summary.json \
    --out_dir results/figures 2>&1 | quiet

say "6/6 cross-validation summary (waits for fold 4)"
wait_done xlite_f4
for f in runs/xlite/fold*/log.csv runs/xlite_k3/fold0/log.csv runs/*/fold0/log.csv; do
  echo "$f $(awk -F, 'NR>1 && $8!="" && $8>b {b=$8; l=$0} END {print l}' $f)"
done | sort -u > results/cv_best_epochs.txt
cat results/cv_best_epochs.txt
du -sh preds results
say "ALL DONE (chain C)"
