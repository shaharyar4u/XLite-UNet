"""Predict a split with sliding-window inference, optional flip-TTA and MC-dropout.

Writes, per case, into --out_dir:
  <id>.nii.gz        BraTS integer labels in the input's voxel space
  <id>_unc.npz       'unc' uint8 (3,H,W,D), value/255 =: normalised predictive entropy per region [0,1]
  <id>_prob.npz      'prob' uint8 (3,H,W,D), value/255 = mean probability (with --save_probs)
  timing.csv         seconds per case

Usage:
  python infer.py --checkpoint runs/xlite/fold0/best.pt --splits splits/splits.json \
      --data_root /data/BraTS2023_GLI/TrainingData --which test --out_dir preds/xlite_test --tta 8 --mc 10 --save_probs
"""
import argparse
import csv
import itertools
import math
import os
import time

import nibabel as nib
import numpy as np
import torch
from monai.inferers import sliding_window_inference

from src.niio import read_nifti
from src.data import cases_for, get_transforms, load_splits, regions_to_labels
from src.model import enable_mc_dropout
from src.utils import device, load_checkpoint

FLIP_SETS = [()] + [c for r in (1, 2, 3) for c in itertools.combinations((2, 3, 4), r)]  # 8 combos


def predict_probs(models, x, patch, tta, mc, amp, overlap=0.5):
    """Return mean probability (3,H,W,D) and normalised binary entropy (3,H,W,D)."""
    flips = FLIP_SETS[: max(1, tta)]
    mean, n = None, 0
    for model in models:
        passes = max(1, mc)
        if mc > 1:
            enable_mc_dropout(model)
        else:
            model.eval()
        for _ in range(passes):
            for f in flips:
                xi = torch.flip(x, f) if f else x
                with torch.no_grad(), torch.autocast(device_type=x.device.type, enabled=amp and x.device.type == "cuda"):
                    lg = sliding_window_inference(xi, patch, 2, model, overlap=overlap)
                p = torch.sigmoid(lg.float())
                p = torch.flip(p, f) if f else p
                mean = p if mean is None else mean + p
                n += 1
    mean = (mean / n)[0]
    eps = 1e-6
    ent = -(mean * torch.log(mean + eps) + (1 - mean) * torch.log(1 - mean + eps)) / math.log(2)
    return mean.cpu().numpy(), ent.clamp(0, 1).cpu().numpy()


def to_u8(a):
    """Store a [0,1] map as uint8 (step 1/255): ~20x smaller on disk than float16."""
    return np.round(np.clip(a, 0, 1) * 255).astype(np.uint8)


def postprocess(prob, et_min_voxels=0, et_label=3, thr=0.5):
    wt, tc, et = prob[0] > thr, prob[1] > thr, prob[2] > thr
    tc &= wt
    et &= tc
    if et_min_voxels and et.sum() < et_min_voxels:
        et[:] = False  # tiny ET is usually a false positive; becomes necrotic core (BraTS practice)
    return regions_to_labels(wt, tc, et, et_label)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", nargs="+", required=True, help="one or more best.pt (several = fold ensemble)")
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--splits", default="splits/splits.json")
    ap.add_argument("--which", default="test", help="test | val | all")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--tta", type=int, default=1, help="number of flip combinations (1 = none, 8 = all)")
    ap.add_argument("--mc", type=int, default=1, help="MC-dropout passes (1 = off)")
    ap.add_argument("--et_min_voxels", type=int, default=0)
    ap.add_argument("--et_label", type=int, default=None, help="3 for BraTS 2023+, 4 for 2021 (default: from splits)")
    ap.add_argument("--save_probs", action="store_true")
    ap.add_argument("--no_unc", action="store_true", help="skip saving uncertainty maps (saves disk for baselines)")
    ap.add_argument("--no_amp", action="store_true")
    a = ap.parse_args()

    dev = device()
    models, cfg = [], None
    for ck in a.checkpoint:
        m, cfg = load_checkpoint(ck, dev)
        models.append(m.to(dev).eval())
    patch = tuple(cfg["patch"])
    et_label = a.et_label
    if et_label is None:
        et_label = load_splits(a.splits).get("et_label", 3) if os.path.exists(a.splits) else 3

    cases = cases_for(a.splits, a.data_root, a.which, a.fold)
    tf = get_transforms("infer")
    os.makedirs(a.out_dir, exist_ok=True)
    tlog = open(os.path.join(a.out_dir, "timing.csv"), "w", newline="")
    tw = csv.writer(tlog)
    tw.writerow(["id", "seconds"])
    for c in cases:
        d = tf({"image": c["image"]})
        x = d["image"].unsqueeze(0).to(dev)
        if dev.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        prob, unc = predict_probs(models, x, patch, a.tta, a.mc, not a.no_amp)
        if dev.type == "cuda":
            torch.cuda.synchronize()
        sec = time.time() - t0
        lab = postprocess(prob, a.et_min_voxels, et_label)
        ref = read_nifti(c["image"][0])
        nib.save(nib.Nifti1Image(lab, ref.affine, ref.header), os.path.join(a.out_dir, f"{c['id']}.nii.gz"))
        if not a.no_unc:
            np.savez_compressed(os.path.join(a.out_dir, f"{c['id']}_unc.npz"), unc=to_u8(unc))
        if a.save_probs:
            np.savez_compressed(os.path.join(a.out_dir, f"{c['id']}_prob.npz"), prob=to_u8(prob))
        tw.writerow([c["id"], round(sec, 3)]); tlog.flush()
        print(f"{c['id']}: {sec:.2f}s")
    tlog.close()


if __name__ == "__main__":
    main()
