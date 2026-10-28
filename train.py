"""Train a model on one CV fold (or on all train+val cases with fold=-1).

Usage:
  python train.py --config configs/xlite.yaml --fold 0
  python train.py --config configs/xlite.yaml --fold 0 --set width=24 kernel=3   # ablation
"""
import argparse
import csv
import os
import time

import torch
import torch.nn.functional as F
from monai.data import DataLoader
from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss, DiceFocalLoss
from monai.metrics import DiceMetric

from src.data import REGIONS, cases_for, get_transforms, make_dataset
from src.factory import build_model
from src.model import count_params
from src.utils import device, load_config, save_json, seed_everything


def make_loss(name):
    # batch Dice (as nnU-Net) avoids the constant-loss trap on crops where a region is absent
    kw = dict(sigmoid=True, batch=True, smooth_nr=1e-5, smooth_dr=1e-5)
    if name == "dicefocal":
        return DiceFocalLoss(**kw)
    return DiceCELoss(**kw)


def ds_loss(loss_fn, outputs, target, weights=(1.0, 0.5, 0.25)):
    if not isinstance(outputs, (list, tuple)):
        return loss_fn(outputs, target)
    total, wsum = 0.0, 0.0
    for o, w in zip(outputs, weights):
        t = target if o.shape[2:] == target.shape[2:] else F.interpolate(target, size=o.shape[2:], mode="nearest")
        total = total + w * loss_fn(o, t)
        wsum += w
    return total / wsum


@torch.no_grad()
def validate(model, loader, patch, dev, amp):
    model.eval()
    metric = DiceMetric(include_background=True, reduction="mean_batch", ignore_empty=False)
    for batch in loader:
        x, y = batch["image"].to(dev), batch["label"].to(dev)
        with torch.autocast(device_type=dev.type, enabled=amp and dev.type == "cuda"):
            logits = sliding_window_inference(x, patch, 2, model, overlap=0.5)
        metric((torch.sigmoid(logits.float()) > 0.5).float(), y)
    d = metric.aggregate().cpu().tolist()
    return dict(zip(REGIONS, d))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--set", nargs="*", default=[], help="override config keys: key=value")
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()

    cfg = load_config(a.config, a.set)
    cfg["fold"] = a.fold
    seed_everything(cfg.get("seed", 42) + a.fold)
    dev = device()
    torch.backends.cudnn.benchmark = bool(cfg.get("cudnn_benchmark", True))  # autotune conv kernels (big win for depthwise 3D)
    amp = bool(cfg.get("amp", True))
    patch = tuple(cfg["patch"])
    out = os.path.join(cfg["out_dir"], f"{cfg.get('run_name', cfg['model'])}", f"fold{a.fold}")
    os.makedirs(out, exist_ok=True)
    save_json(cfg, os.path.join(out, "config.json"))

    tr_cases = cases_for(cfg["splits"], cfg["data_root"], "train", a.fold)
    va_cases = cases_for(cfg["splits"], cfg["data_root"], "val", a.fold) if a.fold >= 0 else []
    cache = cfg.get("cache_dir")  # 'ram' keeps a separate in-memory cache for train and val
    tr_ds = make_dataset(tr_cases, get_transforms("train", patch, cfg.get("samples_per_volume", 2)),
                         cache if cache == "ram" else (os.path.join(cache, "train") if cache else None))
    tr_dl = DataLoader(tr_ds, batch_size=cfg.get("batch_size", 2), shuffle=True,
                       num_workers=cfg.get("num_workers", 8), pin_memory=dev.type == "cuda",
                       persistent_workers=cfg.get("num_workers", 8) > 0, drop_last=True)
    va_dl = None
    if va_cases:
        va_ds = make_dataset(va_cases, get_transforms("val"), cache if cache == "ram" else (os.path.join(cache, "val") if cache else None))
        va_dl = DataLoader(va_ds, batch_size=1, num_workers=max(1, cfg.get("num_workers", 8) // 2))

    model = build_model(cfg).to(dev)
    print(f"model={cfg['model']}  params={count_params(model):,}  train={len(tr_cases)}  val={len(va_cases)}  device={dev}")
    loss_fn = make_loss(cfg.get("loss", "dicece"))
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.get("lr", 1e-3), weight_decay=cfg.get("weight_decay", 1e-5))
    epochs = cfg.get("epochs", 300)
    warm = cfg.get("warmup_epochs", 5)
    sched = torch.optim.lr_scheduler.SequentialLR(
        opt,
        [torch.optim.lr_scheduler.LinearLR(opt, 0.01, 1.0, total_iters=max(1, warm)),
         torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs - warm))],
        milestones=[max(1, warm)],
    )
    scaler = torch.amp.GradScaler("cuda", enabled=amp and dev.type == "cuda")

    start, best = 0, -1.0
    last_ck = os.path.join(out, "last.pt")
    if a.resume and os.path.exists(last_ck):
        ck = torch.load(last_ck, map_location=dev, weights_only=False)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"])
        start, best = ck["epoch"] + 1, ck.get("best", -1.0)
        print(f"resumed from epoch {start}")

    log_path = os.path.join(out, "log.csv")
    new_log = not os.path.exists(log_path) or start == 0
    log = open(log_path, "w" if new_log else "a", newline="")
    w = csv.writer(log)
    if new_log:
        w.writerow(["epoch", "lr", "train_loss", "sec", "val_WT", "val_TC", "val_ET", "val_mean"])

    val_every = cfg.get("val_every", 10)
    for ep in range(start, epochs):
        model.train()
        t0, run, n = time.time(), 0.0, 0
        for batch in tr_dl:
            x, y = batch["image"].to(dev, non_blocking=True), batch["label"].to(dev, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=dev.type, enabled=amp and dev.type == "cuda"):
                outs = model(x)
            outs = [o.float() for o in outs] if isinstance(outs, (list, tuple)) else outs.float()
            loss = ds_loss(loss_fn, outs, y)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 12.0)
            scaler.step(opt); scaler.update()
            run += loss.item(); n += 1
        sched.step()
        row = [ep, opt.param_groups[0]["lr"], run / max(1, n), round(time.time() - t0, 1)]
        state = {"model": model.state_dict(), "config": cfg, "epoch": ep}
        if va_dl is not None and ((ep + 1) % val_every == 0 or ep + 1 == epochs):
            d = validate(model, va_dl, patch, dev, amp)
            mean = sum(d.values()) / 3
            row += [d["WT"], d["TC"], d["ET"], mean]
            if mean > best:
                best = mean
                torch.save({**state, "val": d}, os.path.join(out, "best.pt"))
            print(f"ep {ep:4d} loss {row[2]:.4f}  val WT {d['WT']:.4f} TC {d['TC']:.4f} ET {d['ET']:.4f} (best {best:.4f})")
        else:
            row += ["", "", "", ""]
            print(f"ep {ep:4d} loss {row[2]:.4f}  {row[3]}s")
        w.writerow(row); log.flush()
        torch.save({**state, "opt": opt.state_dict(), "sched": sched.state_dict(), "best": best}, last_ck)
    if va_dl is None:  # fold=-1: final model is the last epoch
        torch.save({"model": model.state_dict(), "config": cfg, "epoch": epochs - 1}, os.path.join(out, "best.pt"))
    log.close()
    print("done:", out)


if __name__ == "__main__":
    main()
