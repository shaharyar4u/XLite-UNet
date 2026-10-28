#!/usr/bin/env bash
# End-to-end smoke test on synthetic data (CPU is fine, ~10-20 min). Checks that every script runs.
set -euo pipefail
cd "$(dirname "$0")/.."
T=${1:-/tmp/brats_smoke}
rm -rf "$T"; mkdir -p "$T"
python tests/make_synthetic.py --out "$T/fake/Top" --n 12 --size 64
( cd "$T/fake" && python -m zipfile -c "$T/data.zip" Top )   # test reading straight from a zip
python make_splits.py --data_root "$T/data.zip" --out "$T/splits.json" --test_frac 0.25 --folds 3
cat > "$T/smoke.yaml" <<YAML
run_name: xlite
model: xlite
data_root: $T/data.zip
splits: $T/splits.json
out_dir: $T/runs
cache_dir: ram
width: 8
kernel: 5
blocks: 1
dropout: 0.1
deep_supervision: true
patch: [64, 64, 64]
batch_size: 2
samples_per_volume: 2
epochs: 2
warmup_epochs: 1
lr: 0.001
loss: dicece
amp: true
num_workers: 0
val_every: 1
YAML
python train.py --config "$T/smoke.yaml" --fold 0
CK="$T/runs/xlite/fold0/best.pt"
python infer.py --checkpoint "$CK" --data_root "$T/data.zip" --splits "$T/splits.json" --which test --out_dir "$T/preds" --tta 2 --mc 2 --save_probs
python evaluate.py --pred_dir "$T/preds" --data_root "$T/data.zip" --splits "$T/splits.json" --which test --out "$T/results/xlite_test.csv"
python eval_uncertainty.py --pred_dir "$T/preds" --data_root "$T/data.zip" --splits "$T/splits.json" --which test --dice_csv "$T/results/xlite_test.csv" --out_dir "$T/results/unc"
python explain.py --checkpoint "$CK" --data_root "$T/data.zip" --splits "$T/splits.json" --which test --n_cases 1 --sanity_cases 1 --figures 1 --unc_dir "$T/preds" --out_dir "$T/results/xai"
python stats.py --proposed "$T/results/xlite_test.csv" --baselines "$T/results/xlite_test.csv" --out "$T/results/stats.csv"
python profile_models.py --configs "$T/smoke.yaml" --volume 64 64 64 --reps 1 --out "$T/results/efficiency.csv"
python make_figures.py --efficiency "$T/results/efficiency.csv" --summaries xlite="$T/results/xlite_test_summary.json" --out_dir "$T/results/figures"
echo "SMOKE TEST PASSED -> $T"
