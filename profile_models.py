"""Efficiency table (Table 2 of the paper): parameters, GFLOPs, peak GPU memory, latency.

  * GFLOPs: one forward pass on a single 4x128^3 patch (torch FlopCounterMode; FLOPs = 2 x MACs)
  * peak GPU memory: inference on one patch, and one training step at batch 2
  * latency: full 240x240x155 volume with sliding-window inference (overlap 0.5, no TTA),
    on GPU and on CPU with a fixed thread count (default 8, i.e. a typical workstation)

Usage:
  python profile_models.py --configs configs/xlite.yaml configs/unet3d.yaml configs/segresnet.yaml \
      configs/unetr.yaml configs/swinunetr.yaml --out results/efficiency.csv
"""
import argparse
import time

import pandas as pd
import torch
from monai.inferers import sliding_window_inference
from torch.utils.flop_counter import FlopCounterMode

from src.factory import build_model
from src.model import count_params
from src.utils import load_config


def timed(fn, dev, reps):
    fn()  # warm-up
    if dev.type == "cuda":
        torch.cuda.synchronize()
    t = time.time()
    for _ in range(reps):
        fn()
    if dev.type == "cuda":
        torch.cuda.synchronize()
    return (time.time() - t) / reps


def main():
    torch.backends.cudnn.benchmark = True  # same autotuned kernels as training
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="+", required=True)
    ap.add_argument("--volume", nargs=3, type=int, default=[240, 240, 155])
    ap.add_argument("--cpu_threads", type=int, default=8)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--skip_cpu", action="store_true")
    ap.add_argument("--set", nargs="*", default=[], help="override config keys for every model")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    rows = []
    for path in a.configs:
        cfg = load_config(path, a.set)
        patch = tuple(cfg["patch"])
        model = build_model(cfg).eval()
        row = {"model": cfg.get("run_name", cfg["model"]), "params_M": count_params(model) / 1e6}
        x = torch.randn(1, 4, *patch)
        with torch.no_grad(), FlopCounterMode(display=False) as fc:
            model(x)
        row["GFLOPs_patch"] = fc.get_total_flops() / 1e9
        vol = torch.randn(1, 4, *a.volume)

        if torch.cuda.is_available():
            dev = torch.device("cuda")
            m = model.to(dev)
            torch.cuda.reset_peak_memory_stats()
            with torch.no_grad(), torch.autocast("cuda"):
                m(x.to(dev))
            row["peak_mem_infer_GB"] = torch.cuda.max_memory_allocated() / 1e9
            m.train()
            torch.cuda.reset_peak_memory_stats()
            xb = torch.randn(2, 4, *patch, device=dev)
            with torch.autocast("cuda"):
                out = m(xb)
                out = out[0] if isinstance(out, (list, tuple)) else out
            out.float().mean().backward()
            row["peak_mem_train_bs2_GB"] = torch.cuda.max_memory_allocated() / 1e9
            m.zero_grad(set_to_none=True); m.eval()
            vd = vol.to(dev)

            def run_gpu():
                with torch.no_grad(), torch.autocast("cuda"):
                    sliding_window_inference(vd, patch, 2, m, overlap=0.5)
            row["sec_volume_gpu"] = timed(run_gpu, dev, a.reps)
            model = m.cpu()
        if not a.skip_cpu:
            torch.set_num_threads(a.cpu_threads)

            def run_cpu():
                with torch.no_grad():
                    sliding_window_inference(vol, patch, 1, model, overlap=0.5)
            row[f"sec_volume_cpu_{a.cpu_threads}t"] = timed(run_cpu, torch.device("cpu"), max(1, a.reps - 2))
        rows.append(row)
        print({k: round(v, 3) if isinstance(v, float) else v for k, v in row.items()})
    pd.DataFrame(rows).to_csv(a.out, index=False)


if __name__ == "__main__":
    main()
