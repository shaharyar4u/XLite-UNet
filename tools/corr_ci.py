"""Case counts and 95% bootstrap CIs for the uncertainty-Dice Spearman correlations (Table 10),
plus per-region case counts for the explanation scores (Table 8).

  python tools/corr_ci.py
"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

u = pd.read_csv("results/unc_xlite/uncertainty_per_case.csv").set_index("id")
d = pd.read_csv("results/xlite_f0_unc_test.csv").set_index("id")
m = u.join(d, how="inner")
rng = np.random.default_rng(0)
for r in ["WT", "TC", "ET"]:
    v = m[[f"mean_unc_pred_{r}", f"dice_{r}"]].dropna().values
    rho = spearmanr(v[:, 0], v[:, 1]).statistic
    boots = []
    for _ in range(2000):
        s = v[rng.integers(0, len(v), len(v))]
        boots.append(spearmanr(s[:, 0], s[:, 1]).statistic)
    lo, hi = np.nanpercentile(boots, [2.5, 97.5])
    print(f"uncertainty-Dice {r}: n = {len(v)}, rho = {rho:.3f}, 95% CI [{lo:.3f}, {hi:.3f}]")

c = pd.read_csv("results/xai_xlite/cam_scores.csv")
print("explanation cases per region:", c.groupby("region").size().to_dict())
