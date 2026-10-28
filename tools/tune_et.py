"""Choose the ET minimum-volume threshold on VALIDATION predictions (never on the test set).

If a predicted enhancing tumour (label 3) has fewer than T voxels, it is relabelled as necrotic
core (label 1), as in the BraTS 2020 nnU-Net post-processing. Prints mean ET Dice and HD95 per T.

  python tools/tune_et.py --pred_dir preds/xlite_f0_val --data_root <zip> --fold 0
"""
import argparse
import os
import sys

import nibabel as nib
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evaluate import hd95  # noqa: E402
from src.data import cases_for, labels_to_regions  # noqa: E402
from src.niio import read_nifti  # noqa: E402
from src.utils import dice_np  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--pred_dir", required=True)
ap.add_argument("--data_root", required=True)
ap.add_argument("--splits", default="splits/splits.json")
ap.add_argument("--fold", type=int, default=0)
ap.add_argument("--thresholds", type=int, nargs="+", default=[0, 25, 50, 100, 200, 300, 500, 750, 1000])
a = ap.parse_args()

res = {t: ([], []) for t in a.thresholds}
for c in cases_for(a.splits, a.data_root, "val", a.fold):
    gimg = read_nifti(c["label"])
    g_et = labels_to_regions(np.asarray(gimg.dataobj))[2]
    sp = gimg.header.get_zooms()[:3]
    p = np.asarray(nib.load(os.path.join(a.pred_dir, f"{c['id']}.nii.gz")).dataobj)
    n_et = int((p == 3).sum())
    for t in a.thresholds:
        p_et = (p == 3) if n_et >= t else np.zeros_like(g_et)
        res[t][0].append(dice_np(p_et, g_et))
        res[t][1].append(hd95(p_et, g_et, sp))

print(f"{'T (voxels)':>10s}  {'Dice ET':>8s}  {'HD95 ET':>8s}")
best = max(a.thresholds, key=lambda t: np.mean(res[t][0]))
for t in a.thresholds:
    d, h = np.mean(res[t][0]), np.mean(res[t][1])
    print(f"{t:10d}  {d:8.4f}  {h:8.2f}" + ("   <- best Dice" if t == best else ""))
