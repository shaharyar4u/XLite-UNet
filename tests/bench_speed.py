"""Speed micro-benchmark (Section 5.2 of the paper). One GPU, mixed precision, 4x128^3 patches.

  train : one forward + backward + optimiser step at batch 4
  infer : one forward pass at batch 1, no gradients
Variants: cuDNN autotuning off/on, memory format (default vs channels_last_3d), XLite kernel 5 vs 3.

  python tests/bench_speed.py | tee logs/bench_speed.txt
"""
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.factory import build_model  # noqa: E402


def timed(fn, steps=10, warm=3):
    for i in range(steps + warm):
        if i == warm:
            torch.cuda.synchronize(); t0 = time.time()
        fn()
    torch.cuda.synchronize()
    return (time.time() - t0) / steps


def run(cfg, autotune, cl):
    torch.backends.cudnn.benchmark = autotune
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


print("GPU:", torch.cuda.get_device_name(0), "| torch", torch.__version__, "| cuDNN", torch.backends.cudnn.version())
tests = [("xlite k5", {"model": "xlite", "kernel": 5}),
         ("xlite k3", {"model": "xlite", "kernel": 3}),
         ("unet3d", {"model": "unet3d"})]
for name, cfg in tests:
    for autotune, cl in ((False, False), (True, False), (True, True)):
        tr, inf = run(cfg, autotune, cl)
        print(f"{name:9s} autotune={autotune!s:5s} channels_last={cl!s:5s}  "
              f"train {tr:.3f} s/step   infer {inf * 1000:.0f} ms/patch", flush=True)
