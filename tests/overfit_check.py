"""Sanity check: the model must be able to overfit a fixed batch (Dice -> ~1 for WT, TC, ET).
  python tests/overfit_check.py --data_root /path/to/a/few/cases --steps 200
"""
import argparse, os, sys, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from monai.losses import DiceCELoss
from src.data import discover_cases, get_transforms
from src.model import XLiteUNet

ap = argparse.ArgumentParser(); ap.add_argument("--data_root", required=True)
ap.add_argument("--steps", type=int, default=200); ap.add_argument("--width", type=int, default=8)
a = ap.parse_args()
torch.manual_seed(0)
tf = get_transforms("val")
cs = discover_cases(a.data_root)[:2]
ds = [tf({"image": c["image"], "label": c["label"]}) for c in cs]
x = torch.stack([d["image"] for d in ds]); y = torch.stack([d["label"] for d in ds])
pad = [(-s) % 8 for s in x.shape[2:]]
x = torch.nn.functional.pad(x, (0, pad[2], 0, pad[1], 0, pad[0])); y = torch.nn.functional.pad(y, (0, pad[2], 0, pad[1], 0, pad[0]))
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu"); x, y = x.to(dev), y.to(dev)
m = XLiteUNet(width=a.width, blocks=1, deep_supervision=False).to(dev)
opt = torch.optim.AdamW(m.parameters(), 3e-3)
lf = DiceCELoss(sigmoid=True, batch=True, smooth_nr=1e-5, smooth_dr=1e-5)
for s in range(a.steps + 1):
    m.train(); opt.zero_grad(); out = m(x); loss = lf(out, y); loss.backward(); opt.step()
    if s % 25 == 0:
        p = torch.sigmoid(out.detach()) > 0.5
        d = [float(2 * (p[:, i] & (y[:, i] > 0)).sum() / (p[:, i].sum() + y[:, i].sum() + 1e-8)) for i in range(3)]
        print(f"step {s:4d} loss {loss.item():.3f}  Dice WT {d[0]:.3f} TC {d[1]:.3f} ET {d[2]:.3f}", flush=True)
