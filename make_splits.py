"""Create a locked test set (stratified by whole-tumour volume) and 5 CV folds on the rest.

Usage:
  python make_splits.py --data_root /data/BraTS2023_GLI/TrainingData --out splits/splits.json
"""
import argparse

import nibabel as nib
import numpy as np
from tqdm import tqdm

from src.niio import read_nifti
from src.data import discover_cases
from src.utils import save_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--out", default="splits/splits.json")
    ap.add_argument("--test_frac", type=float, default=0.2)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--bins", type=int, default=5, help="volume quantile bins for stratification")
    a = ap.parse_args()

    cases = [c for c in discover_cases(a.data_root) if c["label"]]
    if not cases:
        raise SystemExit(f"No labelled cases found under {a.data_root}")
    print(f"{len(cases)} labelled cases")

    vols, label_values = [], set()
    for c in tqdm(cases, desc="reading labels"):
        seg = np.asarray(read_nifti(c["label"]).dataobj)
        label_values |= set(np.unique(seg).tolist())
        vols.append(int((seg > 0).sum()))
    vols = np.array(vols)
    print("label values present:", sorted(label_values),
          "-> ET label is", 4 if 4 in label_values else 3)

    rng = np.random.default_rng(a.seed)
    edges = np.quantile(vols, np.linspace(0, 1, a.bins + 1)[1:-1])
    strata = np.digitize(vols, edges)

    test_idx, rest_idx = [], []
    for s in np.unique(strata):
        idx = rng.permutation(np.where(strata == s)[0])
        n_test = int(round(len(idx) * a.test_frac))
        test_idx += idx[:n_test].tolist()
        rest_idx += idx[n_test:].tolist()

    # folds: stratified round-robin over the remaining cases
    fold_of, k = {}, 0  # k keeps counting across strata so folds stay balanced
    for s in np.unique(strata):
        idx = [i for i in rng.permutation(rest_idx) if strata[i] == s]
        for i in idx:
            fold_of[i] = k % a.folds
            k += 1
    folds = []
    for f in range(a.folds):
        val = sorted(cases[i]["id"] for i in rest_idx if fold_of[i] == f)
        train = sorted(cases[i]["id"] for i in rest_idx if fold_of[i] != f)
        folds.append({"train": train, "val": val})

    out = {
        "seed": a.seed,
        "n_cases": len(cases),
        "et_label": 4 if 4 in label_values else 3,
        "test": sorted(cases[i]["id"] for i in test_idx),
        "trainval": sorted(cases[i]["id"] for i in rest_idx),
        "folds": folds,
    }
    save_json(out, a.out)
    print(f"test={len(out['test'])}  trainval={len(out['trainval'])}  "
          f"fold sizes={[len(f['val']) for f in folds]}  -> {a.out}")


if __name__ == "__main__":
    main()
