"""Bridge to nnU-Net v2 so the strongest baseline uses exactly our split.

1) Export (trainval cases as training data, our 5 folds as splits_final.json, test cases as imagesTs):
   python tools/nnunet_bridge.py export --data_root /data/BraTS2023_GLI/TrainingData \
       --splits splits/splits.json --nnunet_raw $nnUNet_raw --dataset_id 501
   then:  nnUNetv2_plan_and_preprocess -d 501 --verify_dataset_integrity
          cp splits_final.json $nnUNet_preprocessed/Dataset501_BraTSGLI/   (printed by this script)
          nnUNetv2_train 501 3d_fullres 0   (... folds 1-4)
          nnUNetv2_predict -i $nnUNet_raw/Dataset501_BraTSGLI/imagesTs -o preds/nnunet_raw -d 501 -c 3d_fullres
2) Convert predictions back to BraTS labels, then score with evaluate.py:
   python tools/nnunet_bridge.py import --pred_dir preds/nnunet_raw --out_dir preds/nnunet_test --et_label 3

Labels are remapped as in nnU-Net's own BraTS conversion so region training works:
  ED -> 1, NCR -> 2, ET -> 3; regions WT = (1,2,3), TC = (2,3), ET = (3).
"""
import argparse
import json
import os
import shutil

import nibabel as nib
import numpy as np

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data import cases_for, load_splits  # noqa: E402


def export(a):
    if a.data_root.endswith(".zip"):
        raise SystemExit("nnU-Net needs extracted files: point --data_root at an extracted folder, not the zip.")
    name = f"Dataset{a.dataset_id:03d}_BraTSGLI"
    root = os.path.join(a.nnunet_raw, name)
    for sub in ("imagesTr", "labelsTr", "imagesTs"):
        os.makedirs(os.path.join(root, sub), exist_ok=True)
    s = load_splits(a.splits)

    def put_images(c, sub):
        for k, p in enumerate(c["image"]):
            dst = os.path.join(root, sub, f"{c['id']}_{k:04d}.nii.gz")
            if not os.path.exists(dst):
                (os.symlink if a.symlink else shutil.copy)(os.path.abspath(p), dst)

    for c in cases_for(a.splits, a.data_root, "trainval"):
        put_images(c, "imagesTr")
        img = nib.load(c["label"])
        seg = np.asarray(img.dataobj).astype(np.uint8)
        new = np.zeros_like(seg)
        new[seg == 2] = 1
        new[seg == 1] = 2
        new[(seg == 3) | (seg == 4)] = 3
        nib.save(nib.Nifti1Image(new, img.affine, img.header), os.path.join(root, "labelsTr", f"{c['id']}.nii.gz"))
    for c in cases_for(a.splits, a.data_root, "test"):
        put_images(c, "imagesTs")

    ds = {
        "channel_names": {"0": "T1", "1": "T1ce", "2": "T2", "3": "FLAIR"},
        "labels": {"background": 0, "whole tumor": [1, 2, 3], "tumor core": [2, 3], "enhancing tumor": [3]},
        "regions_class_order": [1, 2, 3],
        "numTraining": len(s["trainval"]),
        "file_ending": ".nii.gz",
    }
    with open(os.path.join(root, "dataset.json"), "w") as f:
        json.dump(ds, f, indent=2)
    with open("splits_final.json", "w") as f:
        json.dump(s["folds"], f, indent=2)
    print(f"wrote {root}\ncopy ./splits_final.json into $nnUNet_preprocessed/{name}/ after plan_and_preprocess")


def import_preds(a):
    os.makedirs(a.out_dir, exist_ok=True)
    n = 0
    for fn in sorted(os.listdir(a.pred_dir)):
        if not fn.endswith(".nii.gz"):
            continue
        img = nib.load(os.path.join(a.pred_dir, fn))
        p = np.asarray(img.dataobj).astype(np.uint8)
        out = np.zeros_like(p)
        out[p == 1] = 2
        out[p == 2] = 1
        out[p == 3] = a.et_label
        nib.save(nib.Nifti1Image(out, img.affine, img.header), os.path.join(a.out_dir, fn))
        n += 1
    print(f"converted {n} predictions -> {a.out_dir}")


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    e = sp.add_parser("export")
    e.add_argument("--data_root", required=True)
    e.add_argument("--splits", default="splits/splits.json")
    e.add_argument("--nnunet_raw", required=True)
    e.add_argument("--dataset_id", type=int, default=501)
    e.add_argument("--symlink", action="store_true", help="symlink images instead of copying")
    i = sp.add_parser("import")
    i.add_argument("--pred_dir", required=True)
    i.add_argument("--out_dir", required=True)
    i.add_argument("--et_label", type=int, default=3)
    a = ap.parse_args()
    export(a) if a.cmd == "export" else import_preds(a)


if __name__ == "__main__":
    main()
