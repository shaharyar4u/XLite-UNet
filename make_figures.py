"""Paper figure: mean Dice vs. parameters (bubble size = GFLOPs), and the main results table as CSV.

Usage:
  python make_figures.py --efficiency results/efficiency.csv \
      --summaries xlite=results/xlite_test_summary.json unet3d=results/unet3d_test_summary.json ... \
      --out_dir results/figures
Model names must match the 'model' column of efficiency.csv (the run_name in each config).
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--efficiency", required=True)
    ap.add_argument("--summaries", nargs="+", required=True, help="name=path/to/<run>_summary.json")
    ap.add_argument("--highlight", default="xlite")
    ap.add_argument("--out_dir", required=True)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    eff = pd.read_csv(a.efficiency).set_index("model")
    rows = []
    for item in a.summaries:
        name, path = item.split("=", 1)
        s = json.load(open(path))
        r = {"model": name}
        for k in ("dice_WT", "dice_TC", "dice_ET", "dice_mean", "hd95_WT", "hd95_TC", "hd95_ET", "hd95_mean"):
            r[k] = s[k]["mean"]
            r[k + "_ci"] = f"[{s[k]['ci95'][0]:.3f}, {s[k]['ci95'][1]:.3f}]"
        if name in eff.index:
            r.update(eff.loc[name].to_dict())
        rows.append(r)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(a.out_dir, "main_table.csv"), index=False)

    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    for _, r in df.iterrows():
        if pd.isna(r.get("params_M")):
            continue
        hl = r["model"] == a.highlight
        ax.scatter(r["params_M"], r["dice_mean"], s=30 + 4 * r["GFLOPs_patch"],
                   alpha=0.85 if hl else 0.45, color="#d1495b" if hl else "#4a6fa5", edgecolor="black", lw=0.5)
        ax.annotate(r["model"], (r["params_M"], r["dice_mean"]), xytext=(6, 4),
                    textcoords="offset points", fontsize=8, fontweight="bold" if hl else "normal")
    ax.set_xscale("log")
    ax.set_xlabel("Parameters (millions, log scale)")
    ax.set_ylabel("Mean Dice (WT, TC, ET)")
    ax.set_title("Accuracy vs. model size (bubble area ~ GFLOPs)", fontsize=9)
    ax.grid(alpha=0.3, lw=0.5)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(a.out_dir, f"dice_vs_params.{ext}"), dpi=300)
    print(df.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
