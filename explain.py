"""Explainability analysis with quantitative scores (Table 4 and Fig. 4 of the paper).

For each case (a 128^3 crop centred on the predicted tumour) and each region WT/TC/ET:
  * Seg-Grad-CAM map (Vinogradova et al. 2020) on a chosen decoder layer
      - attribution mass inside the ground-truth region, pointing-game hit
  * Sequence (modality) importance by occlusion: Dice drop when one MRI sequence is zeroed
      - optional Integrated Gradients share per sequence (--ig, slower)
  * Faithfulness: deletion curve AUC (remove top-CAM voxels) vs. random removal
Plus, on the first --sanity_cases cases, the cascading model-parameter randomisation test
(Adebayo et al. 2018): Spearman correlation between the trained CAM and CAMs after
re-initialising layers from the output backwards. Low correlation = the map depends on the model.

Usage:
  python explain.py --checkpoint runs/xlite/fold0/best.pt --data_root ... --splits splits/splits.json \
      --which test --n_cases 50 --out_dir results/xai_xlite --unc_dir preds/xlite_test --figures 6
"""
import argparse
import copy
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from monai.inferers import sliding_window_inference
from scipy.stats import spearmanr

from src.data import MODALITY_NAMES, REGIONS, cases_for, get_transforms
from src.utils import device, load_checkpoint, load_map, trapz

DEL_FRACS = [0.0, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2]


def dice_t(a, b):
    s = a.sum() + b.sum()
    return 1.0 if s == 0 else float(2 * (a & b).sum() / s)


class SegGradCAM:
    def __init__(self, model, layer_name):
        self.model = model
        layer = model.get_submodule(layer_name)
        self.acts = self.grads = None
        self.h = layer.register_forward_hook(self._hook)

    def _hook(self, m, i, o):
        self.acts = o
        if o.requires_grad:
            o.register_hook(lambda g: setattr(self, "grads", g))

    def remove(self):
        self.h.remove()

    def __call__(self, x, region, mask):
        self.model.zero_grad(set_to_none=True)
        x = x.detach().requires_grad_(True)
        with torch.enable_grad():
            logits = self.model(x)
            score = (logits[:, region] * mask).sum()
            score.backward()
        w = self.grads.mean(dim=(2, 3, 4), keepdim=True)
        cam = F.relu((w * self.acts).sum(1, keepdim=True))
        cam = F.interpolate(cam, size=x.shape[2:], mode="trilinear", align_corners=False)[0, 0]
        return (cam / (cam.max() + 1e-8)).detach()


def crop_box(center, shape, p):
    lo = [int(min(max(c - p // 2, 0), max(s - p, 0))) for c, s in zip(center, shape)]
    return tuple(slice(l, l + p) for l in lo)


def predict(model, x):
    with torch.no_grad():
        return torch.sigmoid(model(x)) > 0.5


def deletion_auc(model, x, region, ref, order, brain_idx):
    scores = []
    n = len(brain_idx)
    flat = x.clone().view(1, x.shape[1], -1)
    for f in DEL_FRACS:
        xm = flat.clone()
        k = int(f * n)
        if k:
            xm[:, :, brain_idx[order[:k]]] = 0.0
        p = predict(model, xm.view_as(x))[0, region]
        scores.append(dice_t(p, ref))
    return float(trapz(scores, DEL_FRACS) / DEL_FRACS[-1]), scores


def integrated_gradients(model, x, region, mask, steps):
    from captum.attr import IntegratedGradients

    def f(inp):
        return (model(inp)[:, region] * mask).sum(dim=(1, 2, 3))

    ig = IntegratedGradients(f)
    att = ig.attribute(x, baselines=torch.zeros_like(x), n_steps=steps, internal_batch_size=1)
    s = att.abs().sum(dim=(0, 2, 3, 4))
    return (s / s.sum()).cpu().numpy()


def randomisation_test(model, layer, x, region, mask, cam0, rng):
    names = [n for n, _ in model.named_children() if not n.startswith(("aux", "drop"))]
    m = copy.deepcopy(model)
    idx = rng.choice(cam0.numel(), size=min(20000, cam0.numel()), replace=False)
    ref = cam0.flatten()[idx].cpu().numpy()
    out = []
    for n in reversed(names):
        for sub in m.get_submodule(n).modules():
            if hasattr(sub, "reset_parameters"):
                sub.reset_parameters()
        g = SegGradCAM(m, layer)
        cam = g(x, region, mask)
        g.remove()
        r = spearmanr(ref, cam.flatten()[idx].cpu().numpy()).statistic
        out.append({"randomised_up_to": n, "spearman": float(np.nan_to_num(r))})
    return out


def figure(path, x, gt, pred, cams, unc):
    wt = gt[0].sum(dim=(0, 1)).cpu().numpy()
    z = int(np.argmax(wt))
    flair = x[0, 3, :, :, z].cpu().numpy().T
    t1c = x[0, 1, :, :, z].cpu().numpy().T
    lab = lambda r: (r[0].int() + r[1].int() + r[2].int())[:, :, z].cpu().numpy().T
    panels = [("FLAIR", flair, None), ("T1ce", t1c, None),
              ("Ground truth", flair, lab(gt)), ("Prediction", flair, lab(pred))]
    for name, c in cams.items():
        if c is not None:
            panels.append((f"Seg-Grad-CAM {name}", flair, c[:, :, z].cpu().numpy().T))
    if unc is not None:
        panels.append(("Uncertainty ET", flair, unc[2][:, :, z].T))
    fig, ax = plt.subplots(1, len(panels), figsize=(2.6 * len(panels), 2.8))
    for a, (t, bg, ov) in zip(np.atleast_1d(ax), panels):
        a.imshow(bg, cmap="gray", origin="lower")
        if ov is not None:
            cmap = "viridis" if "Ground" in t or "Pred" in t else "jet"
            a.imshow(np.ma.masked_where(ov <= 0.05, ov), cmap=cmap, alpha=0.55, origin="lower",
                     vmin=0, vmax=3 if cmap == "viridis" else 1)
        a.set_title(t, fontsize=9); a.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--splits", default="splits/splits.json")
    ap.add_argument("--which", default="test")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--n_cases", type=int, default=50)
    ap.add_argument("--layer", default=None, help="Grad-CAM layer (default dec2 for xlite)")
    ap.add_argument("--crop", type=int, default=None, help="crop size (default = training patch)")
    ap.add_argument("--ig", action="store_true", help="also compute Integrated Gradients per sequence")
    ap.add_argument("--ig_steps", type=int, default=16)
    ap.add_argument("--sanity_cases", type=int, default=10)
    ap.add_argument("--figures", type=int, default=6)
    ap.add_argument("--unc_dir", default=None, help="infer.py output dir, to add uncertainty to figures")
    ap.add_argument("--out_dir", required=True)
    a = ap.parse_args()

    dev = device()
    model, cfg = load_checkpoint(a.checkpoint, dev)
    model = model.to(dev).eval()
    layer = a.layer or ("dec2" if cfg["model"] == "xlite" else None)
    if layer is None:
        raise SystemExit("--layer is required for baseline models (a decoder block name, see model.named_modules())")
    p = a.crop or cfg["patch"][0]
    os.makedirs(os.path.join(a.out_dir, "figures"), exist_ok=True)
    rng = np.random.default_rng(0)
    tf = get_transforms("val")
    cam_rows, mod_rows, san_rows = [], [], []

    cases = cases_for(a.splits, a.data_root, a.which, a.fold)[: a.n_cases]
    for ci, c in enumerate(cases):
        d = tf({"image": c["image"], "label": c["label"]})
        X, Y = d["image"].unsqueeze(0).to(dev), d["label"].to(dev)
        pad = [max(0, p - s) for s in X.shape[2:]]
        if any(pad):
            X = F.pad(X, (0, pad[2], 0, pad[1], 0, pad[0]))
            Y = F.pad(Y, (0, pad[2], 0, pad[1], 0, pad[0]))
        with torch.no_grad():
            full = torch.sigmoid(sliding_window_inference(X, (p,) * 3, 1, model, overlap=0.25)) > 0.5
        src = full[0, 0] if full[0, 0].any() else Y[0].bool()
        center = torch.nonzero(src).float().mean(0).long().tolist() if src.any() else [s // 2 for s in X.shape[2:]]
        box = crop_box(center, X.shape[2:], p)
        x = X[(slice(None), slice(None)) + box].contiguous()
        gt = Y[(slice(None),) + box].bool()
        pred = predict(model, x)[0]
        brain = (x[0].abs().sum(0) > 0).flatten()
        brain_idx = torch.nonzero(brain).flatten()

        cams = {}
        for r, rname in enumerate(REGIONS):
            if pred[r].sum() == 0:
                cams[rname] = None
                continue
            g = SegGradCAM(model, layer)
            cam = g(x, r, pred[r].float()[None])
            g.remove()
            cams[rname] = cam
            am = int(torch.argmax(cam))
            row = {"id": c["id"], "region": rname,
                   "mass_in_gt": float((cam * gt[r]).sum() / (cam.sum() + 1e-8)),
                   "mass_in_pred": float((cam * pred[r]).sum() / (cam.sum() + 1e-8)),
                   "pointing_hit": bool(gt[r].flatten()[am]),
                   "region_dice_in_crop": dice_t(pred[r], gt[r])}
            order = torch.argsort(cam.flatten()[brain_idx], descending=True)
            row["deletion_auc_cam"], _ = deletion_auc(model, x, r, pred[r], order, brain_idx)
            rand = torch.from_numpy(rng.permutation(len(brain_idx))).to(dev)
            row["deletion_auc_random"], _ = deletion_auc(model, x, r, pred[r], rand, brain_idx)
            cam_rows.append(row)

            mrow = {"id": c["id"], "region": rname}
            for m, mname in enumerate(MODALITY_NAMES):
                xm = x.clone(); xm[:, m] = 0
                mrow[f"occl_drop_{mname}"] = 1 - dice_t(predict(model, xm)[0, r], pred[r])
            if a.ig:
                for mname, v in zip(MODALITY_NAMES, integrated_gradients(model, x, r, pred[r].float()[None], a.ig_steps)):
                    mrow[f"ig_share_{mname}"] = float(v)
            mod_rows.append(mrow)

            if ci < a.sanity_cases and r == 0:
                for s in randomisation_test(model, layer, x, r, pred[r].float()[None], cam, rng):
                    san_rows.append({"id": c["id"], "region": rname, **s})

        if ci < a.figures:
            unc = None
            if a.unc_dir and os.path.exists(os.path.join(a.unc_dir, f"{c['id']}_unc.npz")):
                u = load_map(os.path.join(a.unc_dir, f"{c['id']}_unc.npz"), "unc")
                u = np.pad(u, [(0, 0)] + [(0, q) for q in pad])
                unc = u[(slice(None),) + box]
            figure(os.path.join(a.out_dir, "figures", f"{c['id']}.png"), x, gt, pred, cams, unc)
        print(f"[{ci + 1}/{len(cases)}] {c['id']}")

    pd.DataFrame(cam_rows).to_csv(os.path.join(a.out_dir, "cam_scores.csv"), index=False)
    pd.DataFrame(mod_rows).to_csv(os.path.join(a.out_dir, "modality_importance.csv"), index=False)
    pd.DataFrame(san_rows).to_csv(os.path.join(a.out_dir, "sanity_randomisation.csv"), index=False)
    if cam_rows:
        print("\nSeg-Grad-CAM scores (mean by region):")
        print(pd.DataFrame(cam_rows).drop(columns="id").groupby("region").mean().round(3).to_string())
        print("\nSequence importance, Dice drop when zeroed (mean by region):")
        print(pd.DataFrame(mod_rows).drop(columns="id").groupby("region").mean().round(3).to_string())
    if san_rows:
        print("\nRandomisation test (mean Spearman vs trained CAM):")
        print(pd.DataFrame(san_rows).groupby("randomised_up_to", sort=False)["spearman"].mean().round(3).to_string())


if __name__ == "__main__":
    main()
