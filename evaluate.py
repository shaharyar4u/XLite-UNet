"""Score predicted label maps against ground truth: Dice and HD95 per region (WT, TC, ET).

BraTS conventions for empty regions: both empty -> Dice 1, HD95 0;
exactly one empty -> Dice 0, HD95 373.13 (the image diagonal used by the challenge).

Usage:
  python evaluate.py --pred_dir preds/xlite_test --data_root /data/BraTS2023_GLI/TrainingData \
      --splits splits/splits.json --which test --out results/xlite_test.csv
Works on any folder of <case_id>.nii.gz predictions (e.g. nnU-Net or DMFNet outputs).
"""
import argparse
import os

import nibabel as nib
import numpy as np
import pandas as pd
import torch
from monai.metrics import compute_hausdorff_distance

from src.niio import read_nifti
from src.data import REGIONS, cases_for, labels_to_regions
from src.utils import dice_np, save_json

HD_MAX = 373.13


def hd95(pred, gt, spacing):
    if pred.sum() == 0 and gt.sum() == 0:
        return 0.0
    if pred.sum() == 0 or gt.sum() == 0:
        return HD_MAX
    p = torch.from_numpy(pred[None, None].astype(np.uint8))
    g = torch.from_numpy(gt[None, None].astype(np.uint8))
    v = compute_hausdorff_distance(p, g, percentile=95, spacing=[float(s) for s in spacing])
    return float(v.item())


def bootstrap_ci(x, n=1000, seed=0):
    x = np.asarray(x, float)
    rng = np.random.default_rng(seed)
    means = rng.choice(x, size=(n, len(x)), replace=True).mean(1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def summarise(df):
    out = {}
    for col in [c for c in df.columns if c != "id"]:
        v = df[col].dropna().values
        lo, hi = bootstrap_ci(v)
        out[col] = {"mean": float(v.mean()), "std": float(v.std(ddof=1)) if len(v) > 1 else 0.0,
                    "median": float(np.median(v)), "ci95": [lo, hi], "n": int(len(v))}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_dir", required=True)
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--splits", default="splits/splits.json")
    ap.add_argument("--which", default="test")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    rows = []
    for c in cases_for(a.splits, a.data_root, a.which, a.fold):
        pf = os.path.join(a.pred_dir, f"{c['id']}.nii.gz")
        if not os.path.exists(pf):
            print("missing prediction:", c["id"]); continue
        gimg = read_nifti(c["label"])
        g = labels_to_regions(np.asarray(gimg.dataobj))
        p = labels_to_regions(np.asarray(nib.load(pf).dataobj))
        sp = gimg.header.get_zooms()[:3]
        r = {"id": c["id"]}
        for i, name in enumerate(REGIONS):
            r[f"dice_{name}"] = dice_np(p[i], g[i])
            r[f"hd95_{name}"] = hd95(p[i], g[i], sp)
        r["dice_mean"] = np.mean([r[f"dice_{n}"] for n in REGIONS])
        r["hd95_mean"] = np.mean([r[f"hd95_{n}"] for n in REGIONS])
        rows.append(r)
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    df.to_csv(a.out, index=False)
    s = summarise(df)
    save_json(s, a.out.replace(".csv", "_summary.json"))
    print(f"{len(df)} cases")
    for k, v in s.items():
        print(f"{k:12s} {v['mean']:.4f} ± {v['std']:.4f}  95% CI [{v['ci95'][0]:.4f}, {v['ci95'][1]:.4f}]")


if __name__ == "__main__":
    main()
