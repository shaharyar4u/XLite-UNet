"""Generate tiny fake BraTS-2023-style cases (ellipsoid 'tumours') for a smoke test of the pipeline.

Not for results: it only checks that every script runs end to end.
  python tests/make_synthetic.py --out /tmp/fake_brats --n 12 --size 64
"""
import argparse
import os

import nibabel as nib
import numpy as np


def ellipsoid(shape, c, r):
    z, y, x = np.ogrid[: shape[0], : shape[1], : shape[2]]
    return ((z - c[0]) / r[0]) ** 2 + ((y - c[1]) / r[1]) ** 2 + ((x - c[2]) / r[2]) ** 2 <= 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--size", type=int, default=64)
    ap.add_argument("--legacy", action="store_true", help="BraTS 2021 naming and ET=4")
    a = ap.parse_args()
    rng = np.random.default_rng(0)
    s = (a.size,) * 3
    for i in range(a.n):
        cid = f"BraTS2021_{i:05d}" if a.legacy else f"BraTS-GLI-{i:05d}-000"
        d = os.path.join(a.out, cid)
        os.makedirs(d, exist_ok=True)
        brain = ellipsoid(s, [a.size / 2] * 3, [a.size * 0.42] * 3)
        c = rng.integers(a.size // 3, 2 * a.size // 3, 3)
        r = rng.uniform(a.size * 0.08, a.size * 0.18)
        wt = ellipsoid(s, c, [r * 1.6] * 3) & brain
        tc = ellipsoid(s, c, [r] * 3) & wt
        nec = ellipsoid(s, c, [r * 0.5] * 3) & tc
        et = tc & ~nec
        lab = np.zeros(s, np.uint8)
        lab[wt] = 2; lab[nec] = 1; lab[et] = 4 if a.legacy else 3
        base = brain * 100.0
        noise = lambda: rng.normal(0, 5, s)
        mods = {
            "t1": base + nec * -40 + noise(),
            "t1c": base + et * 120 + nec * -40 + noise(),
            "t2": base + wt * 60 + noise(),
            "flair": base + wt * 90 + noise(),
        }
        names = (["t1", "t1ce", "t2", "flair"] if a.legacy else ["t1n", "t1c", "t2w", "t2f"])
        sep = "_" if a.legacy else "-"
        aff = np.diag([1.0, 1.0, 1.0, 1.0])
        for key, nm in zip(["t1", "t1c", "t2", "flair"], names):
            img = (mods[key] * brain).astype(np.float32)
            nib.save(nib.Nifti1Image(img, aff), os.path.join(d, f"{cid}{sep}{nm}.nii.gz"))
        nib.save(nib.Nifti1Image(lab, aff), os.path.join(d, f"{cid}{sep}seg.nii.gz"))
    print(f"wrote {a.n} cases to {a.out}")


if __name__ == "__main__":
    main()
