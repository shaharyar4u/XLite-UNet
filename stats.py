"""Paired comparison of the proposed model against baselines on the same cases.

Wilcoxon signed-rank per metric, Holm correction across all tests, and a 95% bootstrap CI
of the mean paired difference (proposed - baseline).

Usage:
  python stats.py --proposed results/xlite_test.csv \
      --baselines results/unet3d_test.csv results/segresnet_test.csv --out results/stats_test.csv
"""
import argparse
import os

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

METRICS = ["dice_WT", "dice_TC", "dice_ET", "hd95_WT", "hd95_TC", "hd95_ET"]


def holm(pvals):
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    m = len(p)
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--proposed", required=True)
    ap.add_argument("--baselines", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_boot", type=int, default=1000)
    a = ap.parse_args()

    prop = pd.read_csv(a.proposed).set_index("id")
    rng = np.random.default_rng(0)
    rows = []
    for bpath in a.baselines:
        base = pd.read_csv(bpath).set_index("id")
        ids = prop.index.intersection(base.index)
        name = os.path.splitext(os.path.basename(bpath))[0]
        for m in METRICS:
            d = (prop.loc[ids, m] - base.loc[ids, m]).values
            try:
                p = wilcoxon(d, zero_method="zsplit").pvalue if np.any(d != 0) else 1.0
            except ValueError:
                p = 1.0
            boots = rng.choice(d, size=(a.n_boot, len(d)), replace=True).mean(1)
            rows.append({"baseline": name, "metric": m, "n": len(ids),
                         "proposed_mean": prop.loc[ids, m].mean(), "baseline_mean": base.loc[ids, m].mean(),
                         "mean_diff": d.mean(), "ci_low": np.percentile(boots, 2.5),
                         "ci_high": np.percentile(boots, 97.5), "p_wilcoxon": p})
    df = pd.DataFrame(rows)
    df["p_holm"] = holm(df["p_wilcoxon"].values)
    df["significant_0.05"] = df["p_holm"] < 0.05
    df.to_csv(a.out, index=False)
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(df.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
