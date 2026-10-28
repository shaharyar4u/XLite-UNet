# XLite-UNet

A parameter-efficient 3D network (448,297 parameters) for glioma sub-region segmentation on BraTS 2023, with explainability and uncertainty evaluation.

This repository contains the code for the manuscript *"XLite-UNet: A Parameter-Efficient 3D Network for Glioma Segmentation with Explainability and Uncertainty Evaluation"* (S. A. Ansari, Bennett University; under review). It covers data splitting, training, inference, evaluation, statistics, explanation and uncertainty analysis, efficiency profiling and figures.

## Main results (internal test set, n = 250, fold-0 models, eight-flip TTA)

| Model | Params (M) | Dice WT | Dice TC | Dice ET | Mean Dice | HD95 WT / TC / ET (mm) |
| --- | --- | --- | --- | --- | --- | --- |
| 3D U-Net | 5.75 | 0.914 | 0.897 | 0.865 | 0.8918 | 7.78 / 8.87 / 10.82 |
| SegResNet | 4.70 | 0.921 | 0.904 | 0.870 | 0.8984 | 8.56 / 8.72 / 10.57 |
| UNETR | 102.47 | 0.923 | 0.891 | 0.855 | 0.8897 | 7.96 / 9.73 / 14.29 |
| Swin UNETR | 62.19 | 0.923 | 0.901 | 0.866 | 0.8966 | 7.78 / 7.61 / 13.51 |
| **XLite-UNet** | **0.45** | 0.922 | 0.903 | 0.868 | 0.8977 | **6.31 / 6.58 / 10.17** |

The test set is an internal 20% split of the public BraTS 2023 training data, not the official challenge test set. See the manuscript for confidence intervals, paired tests, cross-validation, efficiency, explainability and uncertainty results, and limitations.

## Repository layout

```
configs/              one YAML file per model (set data_root before use)
src/                  data pipeline (reads NIfTI directly from the BraTS zip), XLite-UNet, model factory, helpers
splits/               splits.json: the exact test set and five folds used in the paper
make_splits.py        creates splits.json (stratified 20% test set + 5 folds)
train.py              training: AMP, deep supervision, warm-up + cosine schedule, resume
infer.py              sliding-window inference, flip TTA, Monte Carlo dropout, uncertainty maps
evaluate.py           Dice and HD95 per region with BraTS conventions, bootstrap CIs
stats.py              paired Wilcoxon tests, Holm correction, bootstrap CIs of paired differences
explain.py            Seg-Grad-CAM, sequence occlusion, Integrated Gradients, deletion and randomisation tests
eval_uncertainty.py   ECE, reliability diagram, QU-BraTS-style score, uncertainty-Dice correlation
profile_models.py     parameters, GFLOPs, peak GPU memory, GPU and CPU time per volume
make_figures.py       Dice-versus-parameters figure and main table
tools/                ET threshold tuning (tune_et.py), correlation CIs (corr_ci.py), nnU-Net bridge
scripts/              Docker helpers and the unattended pipelines used on the GPU server
tests/                synthetic smoke test, overfitting check, label check, speed micro-benchmark
```

## Installation

Tested with the NVIDIA NGC container `nvcr.io/nvidia/pytorch:26.04-py3` (PyTorch 2.12.0a0, cuDNN 9.21.0) plus MONAI 1.6.1.

Inside that container:

```bash
bash scripts/setup_env.sh        # installs monai, nibabel, captum, einops into $HOME/pylibs
```

Or in your own environment (install PyTorch for your CUDA version first):

```bash
pip install -r requirements.txt
```

## Data

Register for BraTS 2023 on Synapse and download `ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData.zip` (1,251 labelled cases). The data are not included here and may not be redistributed under the BraTS data-use agreement. The code reads cases directly from the zip, so extraction is optional.

Set `data_root` in every file under `configs/` to the path of the zip (or of an extracted folder).

## Reproducing the paper

All commands run from the repository root. `DATA` is the path to the zip.

```bash
# 0. checks (synthetic data, CPU is fine)
bash tests/run_smoke.sh /tmp/brats_smoke
python tests/overfit_check.py --data_root $DATA --steps 200

# 1. splits: use the provided splits/splits.json to reproduce the paper exactly
#    (python make_splits.py --data_root $DATA --out splits/splits.json creates a new one)

# 2. training
for f in 0 1 2 3 4; do python train.py --config configs/xlite.yaml --fold $f; done
python train.py --config configs/xlite_k3.yaml --fold 0                 # kernel-size ablation
for m in unet3d segresnet unetr swinunetr; do python train.py --config configs/$m.yaml --fold 0; done

# 3. test-set inference (eight-flip TTA) and scoring
python infer.py --checkpoint runs/xlite/fold0/best.pt --data_root $DATA --which test \
    --out_dir preds/xlite_f0_test --tta 8 --no_unc
python evaluate.py --pred_dir preds/xlite_f0_test --data_root $DATA --which test \
    --out results/xlite_f0_test.csv
#    repeat for each baseline; for the five-model ensemble pass all five best.pt files to --checkpoint

# 4. paired statistics (24 tests, Holm correction)
python stats.py --proposed results/xlite_f0_test.csv --baselines results/unet3d_test.csv \
    results/segresnet_test.csv results/unetr_test.csv results/swinunetr_test.csv --out results/stats_f0_all.csv

# 5. uncertainty (4 flips x 5 Monte Carlo dropout passes)
python infer.py --checkpoint runs/xlite/fold0/best.pt --data_root $DATA --which test \
    --out_dir preds/xlite_f0_unc --tta 4 --mc 5 --save_probs
python evaluate.py --pred_dir preds/xlite_f0_unc --data_root $DATA --which test --out results/xlite_f0_unc_test.csv
python eval_uncertainty.py --pred_dir preds/xlite_f0_unc --data_root $DATA --which test \
    --dice_csv results/xlite_f0_unc_test.csv --out_dir results/unc_xlite

# 6. explainability (first 50 test cases)
python explain.py --checkpoint runs/xlite/fold0/best.pt --data_root $DATA --which test \
    --n_cases 50 --ig --sanity_cases 10 --figures 6 --unc_dir preds/xlite_f0_unc --out_dir results/xai_xlite
python tools/corr_ci.py

# 7. efficiency and figure
python profile_models.py --configs configs/xlite.yaml configs/xlite_k3.yaml configs/unet3d.yaml \
    configs/segresnet.yaml configs/unetr.yaml configs/swinunetr.yaml --out results/efficiency.csv
python make_figures.py --efficiency results/efficiency.csv --summaries xlite=results/xlite_f0_test_summary.json \
    unet3d=results/unet3d_test_summary.json segresnet=results/segresnet_test_summary.json \
    unetr=results/unetr_test_summary.json swinunetr=results/swinunetr_test_summary.json --out_dir results/figures
```

ET post-processing was tested with `tools/tune_et.py` on fold-0 validation predictions and gave no improvement, so it was not applied.

On a shared Docker GPU server, `scripts/run.sh <gpu> <job_name> <command>` runs any command in the background, and `scripts/pipeline_a.sh`, `pipeline_b.sh` and `pipeline_c.sh` chain the evaluation steps. Set `BRATS_ZIP` if the data are not at `$HOME/data/`.

## Notes

- Training used one NVIDIA H200 GPU per run. With less memory, use `--set batch_size=1` or `--set patch=[96,96,96]`.
- Seeds are fixed (42 + fold), but cuDNN kernels are not forced to be deterministic, so repeated runs can differ slightly.
- `eval_uncertainty.py` implements a QU-BraTS-style score from the published definition. It is not the official challenge code (https://github.com/RagMeh11/QU-BraTS).

## Use of AI tools

The code in this repository was written with the help of Claude (Anthropic), a large language model assistant. The author ran all experiments and is responsible for the code and results.

## Citation

If you use this code, please cite the manuscript (details will be added on publication).

## Licence

MIT (see LICENSE). The BraTS data have their own terms of use.
