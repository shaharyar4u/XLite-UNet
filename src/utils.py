import json
import os
import random

import numpy as np
import torch
import yaml


def load_config(path, overrides=None):
    with open(path) as f:
        cfg = yaml.safe_load(f)
    for item in overrides or []:  # key=value pairs from the command line
        k, v = item.split("=", 1)
        cfg[k] = yaml.safe_load(v)
    return cfg


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_json(obj, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def dice_np(a, b):
    """Dice for boolean arrays with the BraTS empty-case convention (both empty -> 1)."""
    a, b = a.astype(bool), b.astype(bool)
    s = a.sum() + b.sum()
    if s == 0:
        return 1.0
    return 2.0 * np.logical_and(a, b).sum() / s


def load_checkpoint(path, map_location="cpu"):
    from .factory import build_model

    ck = torch.load(path, map_location=map_location, weights_only=False)
    model = build_model(ck["config"])
    model.load_state_dict(ck["model"])
    return model, ck["config"]


def trapz(y, x):
    """np.trapezoid on NumPy 2, np.trapz on NumPy 1."""
    f = getattr(np, "trapezoid", None) or getattr(np, "trapz")
    return f(y, x)


def load_map(path, key):
    """Load a probability/uncertainty map saved by infer.py (uint8 = value*255; float16 also accepted)."""
    a = np.load(path)[key]
    return a.astype(np.float32) / 255.0 if a.dtype == np.uint8 else a.astype(np.float32)
