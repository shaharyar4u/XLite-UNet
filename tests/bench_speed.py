"""Speed checks on one GPU (AMP, 4x128^3 patches, cuDNN autotuning on).

  train : one fwd+bwd+step at batch 4
  infer : one forward pass at batch 1, no grad
Variants: memory format (default vs channels_last_3d) and XLite kernel size 5 vs 3.

  python tests/bench_speed.py
"""
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.factory import build_model  # noqa: E402

torch.backends.cudnn.benchmark = True


def timed(fn, steps=10, warm=3):
    for i in range(steps + warm):
        if i == warm:
            torch.cuda.synchronize(); t0 = time.time()
        fn()
    torch.cuda.synchronize()
    return (time.time() - t0) / steps


def run(cfg, cl):
    fmt = torch.channels_last_3d if cl else torch.contiguous_format
    m = build_model({"patch": [128, 128, 128], **cfg}).cuda().to(memory_format=fmt)
    opt = torch.optim.AdamW(m.parameters(), 1e-3)
    xt = torch.randn(4, 4, 128, 128, 128, device="cuda").to(memory_format=fmt)
    xi = xt[:1].contiguous().to(memory_format=fmt)

    def train():
        m.train()
        with torch.autocast("cuda"):
            out = m(xt)
            out = out[0] if isinstance(out, (list, tuple)) else out
            loss = out.float().mean()
        opt.zero_grad(); loss.backward(); opt.step()

    def infer():
        m.eval()
        with torch.no_grad(), torch.autocast("cuda"):
            m(xi)

    return timed(train), timed(infer)


tests = [
    ("xlite k5", {"model": "xlite", "kernel": 5}),
    ("xlite k3", {"model": "xlite", "kernel": 3}),
    ("unet3d", {"model": "unet3d"}),
]
for name, cfg in tests:
    for cl in (False, True):
        tr, inf = run(cfg, cl)
        print(f"{name:9s} channels_last={cl!s:5s}  train {tr:.3f} s/step   infer {inf * 1000:.0f} ms/patch", flush=True)
