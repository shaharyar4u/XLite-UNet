"""Uncertainty and calibration analysis (Table 5 and Fig. 5 of the paper).

Needs infer.py outputs made with --save_probs (and --tta/--mc for meaningful uncertainty).
Per case and region:
  * ECE (15 bins) of voxel probabilities inside the brain
  * QU-BraTS-style score: filter voxels whose uncertainty exceeds tau in {1.0, .75, .5, .25};
    score = (AUC_Dice + (1 - AUC_filteredTP) + (1 - AUC_filteredTN)) / 3   (Mehta et al. 2022)
    NOTE: this is a re-implementation for development; report final numbers with the
    official code at https://github.com/RagMeh11/QU-BraTS
  * mean uncertainty inside the predicted region, later correlated with Dice (Spearman)
Also writes reliability-diagram bins pooled over all cases.

Usage:
  python eval_uncertainty.py --pred_dir preds/xlite_test --data_root ... --splits splits/splits.json \
      --which test --dice_csv results/xlite_test.csv --out_dir results/unc_xlite
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from src.niio import read_nifti
from src.data import REGIONS, cases_for, labels_to_regions
from src.utils import dice_np, load_map, save_json, trapz

TAUS = [1.0, 0.75, 0.5, 0.25]
NB = 15


def qu_brats(pred, gt, unc):
    tp0 = (pred & gt).sum()
    tn0 = (~pred & ~gt).sum()
    dsc, ftp, ftn = [], [], []
    for t in TAUS:
        keep = unc <= t
        p, g = pred & keep, gt & keep
        dsc.append(dice_np(p, g))
        ftp.append(0.0 if tp0 == 0 else (tp0 - (p & g).sum()) / tp0)
        ftn.append(0.0 if tn0 == 0 else (tn0 - (~pred & ~gt & keep).sum()) / tn0)
    x = np.array(TAUS)[::-1]  # increasing threshold for the AUC
    auc = lambda v: float(trapz(np.array(v)[::-1], x) / (x[-1] - x[0]))
    a_d, a_tp, a_tn = auc(dsc), auc(ftp), auc(ftn)
    return (a_d + (1 - a_tp) + (1 - a_tn)) / 3, a_d, a_tp, a_tn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_dir", required=True)
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--splits", default="splits/splits.json")
    ap.add_argument("--which", default="test")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--dice_csv", default=None, help="evaluate.py CSV, for the uncertainty-Dice correlation")
    ap.add_argument("--out_dir", required=True)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    bins_cnt = np.zeros((3, NB)); bins_conf = np.zeros((3, NB)); bins_acc = np.zeros((3, NB))
    edges = np.linspace(0, 1, NB + 1)
    rows = []
    for c in cases_for(a.splits, a.data_root, a.which, a.fold):
        pf, uf = (os.path.join(a.pred_dir, f"{c['id']}_{s}.npz") for s in ("prob", "unc"))
        if not (os.path.exists(pf) and os.path.exists(uf)):
            print("skip (no prob/unc):", c["id"]); continue
        prob = load_map(pf, "prob")
        unc = load_map(uf, "unc")
        gt = labels_to_regions(np.asarray(read_nifti(c["label"]).dataobj))
        brain = np.asarray(read_nifti(c["image"][3]).dataobj) != 0
        row = {"id": c["id"]}
        for r, name in enumerate(REGIONS):
            pr, g, u = prob[r][brain], gt[r][brain], unc[r][brain]
            b = np.clip(np.digitize(pr, edges) - 1, 0, NB - 1)
            cnt = np.bincount(b, minlength=NB)
            conf = np.bincount(b, weights=pr, minlength=NB)
            acc = np.bincount(b, weights=g.astype(float), minlength=NB)
            bins_cnt[r] += cnt; bins_conf[r] += conf; bins_acc[r] += acc
            nz = cnt > 0
            row[f"ece_{name}"] = float(np.sum(np.abs(acc[nz] - conf[nz])) / cnt.sum())
            pred = pr > 0.5
            s, ad, atp, atn = qu_brats(pred, g, u)
            row[f"qubrats_{name}"], row[f"auc_dice_{name}"] = s, ad
            row[f"auc_ftp_{name}"], row[f"auc_ftn_{name}"] = atp, atn
            row[f"mean_unc_pred_{name}"] = float(u[pred].mean()) if pred.any() else np.nan
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(a.out_dir, "uncertainty_per_case.csv"), index=False)

    summary = {k: float(df[k].mean()) for k in df.columns if k != "id"}
    if a.dice_csv and os.path.exists(a.dice_csv) and len(df) > 2:
        dd = pd.read_csv(a.dice_csv).set_index("id")
        m = df.set_index("id").join(dd, how="inner")
        for name in REGIONS:
            v = m[[f"mean_unc_pred_{name}", f"dice_{name}"]].dropna()
            if len(v) > 2:
                summary[f"spearman_unc_vs_dice_{name}"] = float(spearmanr(v.iloc[:, 0], v.iloc[:, 1]).statistic)
    save_json(summary, os.path.join(a.out_dir, "uncertainty_summary.json"))

    rel = []
    fig, ax = plt.subplots(1, 3, figsize=(9, 3))
    for r, name in enumerate(REGIONS):
        nz = bins_cnt[r] > 0
        conf = bins_conf[r][nz] / bins_cnt[r][nz]; acc = bins_acc[r][nz] / bins_cnt[r][nz]
        for cf, ac, n in zip(conf, acc, bins_cnt[r][nz]):
            rel.append({"region": name, "confidence": cf, "accuracy": ac, "voxels": int(n)})
        ax[r].plot([0, 1], [0, 1], "--", color="grey", lw=1)
        ax[r].plot(conf, acc, "o-", ms=3)
        ax[r].set_title(f"{name} (ECE {summary.get(f'ece_{name}', float('nan')):.3f})", fontsize=9)
        ax[r].set_xlabel("Predicted probability"); ax[r].set_ylabel("Observed frequency")
    fig.tight_layout(); fig.savefig(os.path.join(a.out_dir, "reliability.png"), dpi=300); plt.close(fig)
    pd.DataFrame(rel).to_csv(os.path.join(a.out_dir, "reliability_bins.csv"), index=False)
    for k, v in summary.items():
        print(f"{k:28s} {v:.4f}")


if __name__ == "__main__":
    main()
