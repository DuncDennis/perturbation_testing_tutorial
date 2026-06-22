"""Raised-cosine Brain-FID: one figure per light intensity, 4 bars
(LIF vs lowBio rank 1/2/3). Bars = perturbation Brain-FID averaged over opto
injection scale and tau (error bars = std); dashed lines = free-running test FID.
"""
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = __import__("pathlib").Path(__file__).parent
pert = pd.read_csv(HERE / "perturbation_metrics.csv")
summ = pd.read_csv(HERE / "metrics_summary.csv")

PLAIN, LOWR = (sys.argv[1], sys.argv[2]) if len(sys.argv) > 2 else ("lif", "lowBio")
MODE = sys.argv[3] if len(sys.argv) > 3 else "freerun"
RANKS = [1, 2, 3]
LEVELS = [1.3, 1.7, 2.0]
LIF_C = "#7f7f7f"
RANK_C = {1: "#94d2bd", 2: "#2a9d8f", 3: "#1d6f63"}   # lowBio teal gradient

plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update({"font.size": 11, "axes.titlesize": 13, "axes.edgecolor": "#444"})

fs = summ[(summ["mode"] == MODE) & (summ["constraint"] == "dale")]
lif_test = fs[fs["model"] == PLAIN]["Brain_FID"].mean()
low_test = {r: fs[(fs["model"] == LOWR) & (fs["rank"] == r)]["Brain_FID"].mean() for r in RANKS}

base = pert[(pert["mode"] == MODE) & (pert["constraint"] == "dale")
            & (pert["perturbation"] == "raised_cosine")]
lif = base[base["model"] == PLAIN]
low = base[base["model"] == LOWR]

labels = [f"{PLAIN}"] + [f"{LOWR}\nrank {r}" for r in RANKS]
colors = [LIF_C] + [RANK_C[r] for r in RANKS]
tests = [lif_test] + [low_test[r] for r in RANKS]

for lvl in LEVELS:
    ld = lif[lif["light"] == lvl]["Brain_FID"]
    means = [ld.mean()]
    stds = [ld.std()]
    for r in RANKS:
        d = low[(low["light"] == lvl) & (low["rank"] == r)]["Brain_FID"]
        means.append(d.mean())
        stds.append(d.std())

    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    x = np.arange(4)
    ax.bar(x, means, 0.62, yerr=stds, capsize=4, color=colors,
           edgecolor="black", linewidth=0.8)
    for xi, t, c in zip(x, tests, colors):
        ax.hlines(t, xi - 0.31, xi + 0.31, color=c, ls="--", lw=2.2)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Brain-FID")
    ax.set_title(f"[{MODE}]  raised_cosine, light l = {lvl}\n"
                 f"dale  ·  bars = perturbation (mean ± std over opto scale & τ)")
    ax.set_ylim(bottom=0)
    ax.legend(handles=[Line2D([0], [0], color="k", ls="--", lw=2,
                              label="test (no pert.), matched colour")],
              loc="upper left", frameon=True)
    fig.tight_layout()
    out = HERE / f"barplot_cosine_{MODE}_{PLAIN}_{LOWR}_allranks_l{lvl}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print("wrote", out)
