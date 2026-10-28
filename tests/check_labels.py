"""Find cases whose tumour label is empty, raw and after the cached training transforms.

  python tests/check_labels.py --data_root <zip or folder> --splits splits/splits.json
"""
import argparse
import os
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from monai import transforms as T  # noqa: E402
from src.data import cases_for, get_transforms  # noqa: E402
from src.niio import read_nifti  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--data_root", required=True)
ap.add_argument("--splits", default="splits/splits.json")
ap.add_argument("--n_transform", type=int, default=30)
a = ap.parse_args()

cases = cases_for(a.splits, a.data_root, "trainval")


def raw(c):
    seg = np.asarray(read_nifti(c["label"]).dataobj)
    return c["id"], int((seg > 0).sum()), sorted(np.unique(seg).tolist())


with ThreadPoolExecutor(16) as ex:
    res = list(ex.map(raw, cases))
empty = [r for r in res if r[1] == 0]
print(f"raw labels: {len(res)} cases, {len(empty)} with NO tumour voxels")
for r in empty[:20]:
    print("   empty:", r[0], "values", r[2])
vols = np.array([r[1] for r in res])
print("tumour voxels per case: min", vols.min(), "median", int(np.median(vols)), "max", vols.max())

tf = get_transforms("train", (128, 128, 128), 2)
first = [i for i, t in enumerate(tf.transforms) if isinstance(t, T.Randomizable)][0]
bad = 0
for c in cases[: a.n_transform]:
    d = c
    for t in tf.transforms[:first]:
        d = t(d)
    lab = d["label"]
    fg = int(((lab.sum(0) > 0) & (d["image"] > 0).any(0)).sum())
    if fg == 0:
        bad += 1
        print("   after transforms EMPTY:", c["id"], "label sums", lab.sum(dim=(1, 2, 3)).tolist())
print(f"after cached transforms: {bad}/{a.n_transform} checked cases have no usable foreground")
