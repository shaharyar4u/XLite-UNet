"""Data discovery, region conversion and MONAI transform pipelines for BraTS.

Supports both naming schemes:
  BraTS 2023+:  <case>/<case>-t1n.nii.gz, -t1c, -t2w, -t2f, -seg   (labels 1=NCR, 2=ED, 3=ET)
  BraTS 2021:   <case>/<case>_t1.nii.gz, _t1ce, _t2, _flair, _seg (labels 1=NCR, 2=ED, 4=ET)
Channel order is always [T1, T1ce, T2, FLAIR].
"""
import glob
import json
import os

import numpy as np
import torch
from monai import transforms as T
from monai.data import CacheDataset, Dataset, PersistentDataset

from .niio import SEP, read_nifti, zip_members

SEQ_2023 = ["t1n", "t1c", "t2w", "t2f"]
SEQ_2021 = ["t1", "t1ce", "t2", "flair"]
MODALITY_NAMES = ["T1", "T1ce", "T2", "FLAIR"]
REGIONS = ["WT", "TC", "ET"]


def _match(case_dir, suffix):
    hits = glob.glob(os.path.join(case_dir, f"*-{suffix}.nii*")) + glob.glob(
        os.path.join(case_dir, f"*_{suffix}.nii*")
    )
    return sorted(hits)[0] if hits else None


def find_case(case_dir):
    """Return {'id', 'image': [4 paths], 'label': path|None} or None if not a case folder."""
    for seqs in (SEQ_2023, SEQ_2021):
        imgs = [_match(case_dir, s) for s in seqs]
        if all(imgs):
            return {
                "id": os.path.basename(os.path.normpath(case_dir)),
                "image": imgs,
                "label": _match(case_dir, "seg"),
            }
    return None


def _cases_in_zip(zpath):
    """Group the NIfTI members of a zip archive by folder and build case dicts."""
    by_dir = {}
    for m in zip_members(zpath):
        by_dir.setdefault(os.path.dirname(m), []).append(m)
    cases = []
    for d, files in by_dir.items():
        def match(suffix):
            hits = sorted(f for f in files if os.path.basename(f).split(".nii")[0].endswith((f"-{suffix}", f"_{suffix}")))
            return f"{zpath}{SEP}{hits[0]}" if hits else None
        for seqs in (SEQ_2023, SEQ_2021):
            imgs = [match(s) for s in seqs]
            if all(imgs):
                cases.append({"id": os.path.basename(d), "image": imgs, "label": match("seg")})
                break
    return sorted(cases, key=lambda c: c["id"])


def discover_cases(root):
    """Return every BraTS case under `root` (a folder, or a .zip read without extracting), sorted by id."""
    if os.path.isfile(root) and root.endswith(".zip"):
        return _cases_in_zip(root)
    cases = []
    for dirpath, _, files in os.walk(root):
        if any(f.endswith((".nii", ".nii.gz")) for f in files):
            c = find_case(dirpath)
            if c:
                cases.append(c)
    return sorted(cases, key=lambda c: c["id"])


def labels_to_regions(lab):
    """Integer label map (any shape) -> stacked [WT, TC, ET] boolean regions.

    Works for both label conventions because ET is 3 in 2023+ and 4 in 2021.
    """
    et = (lab == 3) | (lab == 4)
    tc = et | (lab == 1)
    wt = lab > 0
    if isinstance(lab, torch.Tensor):
        return torch.stack([wt, tc, et], 0)
    return np.stack([wt, tc, et], 0)


def regions_to_labels(wt, tc, et, et_label=3):
    """Nested boolean regions -> BraTS integer label map (uint8)."""
    out = np.zeros(wt.shape, dtype=np.uint8)
    out[wt] = 2
    out[tc & wt] = 1
    out[et & tc & wt] = et_label
    return out


class ConvertToRegionsd(T.MapTransform):
    """(1,H,W,D) integer label -> (3,H,W,D) float [WT, TC, ET]."""

    def __call__(self, data):
        d = dict(data)
        for k in self.key_iterator(d):
            lab = d[k]
            lab = lab[0] if lab.ndim == 4 else lab
            d[k] = labels_to_regions(lab).float()
        return d


class LoadCased(T.MapTransform):
    """Load the 4 sequences as a (4,H,W,D) float tensor and the label as (1,H,W,D).

    Uses nibabel directly so the same code reads plain files and members of a .zip archive.
    """

    def __init__(self):
        super().__init__(keys=["image", "label"], allow_missing_keys=True)

    def __call__(self, data):
        d = dict(data)
        d["image"] = torch.from_numpy(
            np.stack([np.asarray(read_nifti(p).dataobj, dtype=np.float32) for p in d["image"]], 0))
        if d.get("label"):
            d["label"] = torch.from_numpy(np.asarray(read_nifti(d["label"]).dataobj).astype(np.float32))[None]
        else:
            d.pop("label", None)
        return d


def get_transforms(mode, patch=(128, 128, 128), samples_per_volume=2):
    load = [
        LoadCased(),
        ConvertToRegionsd(keys=["label"], allow_missing_keys=True),
        T.NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
    ]
    if mode in ("val", "infer"):
        return T.Compose(load)
    if mode != "train":
        raise ValueError(mode)
    return T.Compose(
        load
        + [
            T.CropForegroundd(keys=["image", "label"], source_key="image", allow_smaller=True),
            T.SpatialPadd(keys=["image", "label"], spatial_size=patch),
            # --- everything above is deterministic and cached by PersistentDataset ---
            T.RandCropByPosNegLabeld(
                keys=["image", "label"], label_key="label", spatial_size=patch,
                pos=1, neg=1, num_samples=samples_per_volume, image_key="image", image_threshold=0,
            ),
            T.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
            T.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
            T.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
            T.RandAffined(
                keys=["image", "label"], prob=0.2, rotate_range=(0.26, 0.26, 0.26),
                scale_range=(0.1, 0.1, 0.1), mode=("bilinear", "nearest"), padding_mode="zeros",
            ),
            T.RandScaleIntensityd(keys="image", factors=0.1, prob=0.5),
            T.RandShiftIntensityd(keys="image", offsets=0.1, prob=0.5),
            T.RandGaussianNoised(keys="image", prob=0.1, std=0.1),
        ]
    )


def load_splits(path):
    with open(path) as f:
        return json.load(f)


def cases_for(split_file, data_root, which, fold=0):
    """which in {'train','val','test','trainval','all'}; fold=-1 means use all non-test cases for training."""
    all_cases = {c["id"]: c for c in discover_cases(data_root)}
    if which == "all":
        return list(all_cases.values())
    s = load_splits(split_file)
    if which == "test":
        ids = s["test"]
    elif which == "trainval":
        ids = s["trainval"]
    elif fold < 0:
        ids = s["trainval"] if which == "train" else []
    else:
        ids = s["folds"][fold][which]
    missing = [i for i in ids if i not in all_cases]
    if missing:
        raise FileNotFoundError(f"{len(missing)} split ids not found under {data_root}, e.g. {missing[:3]}")
    return [all_cases[i] for i in ids]


def make_dataset(cases, transforms, cache_dir=None, cache_workers=16):
    """cache_dir: None = no cache, 'ram' = keep preprocessed volumes in memory, a path = disk cache."""
    if cache_dir == "ram":
        return CacheDataset(cases, transform=transforms, cache_rate=1.0, num_workers=cache_workers)
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        return PersistentDataset(cases, transform=transforms, cache_dir=cache_dir)
    return Dataset(cases, transform=transforms)
