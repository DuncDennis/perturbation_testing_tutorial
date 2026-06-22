"""Brain-FID barplots: dale-LIF vs dale-lowBio, one figure per tau.

Per figure (a given tau):
  - panels for each perturbation waveform (pulse, fast_pulses, raised_cosine):
    x = light level, bars = perturbation Brain-FID (LIF gray, lowBio coloured),
    averaged over opto injection scale (and over rank for lowBio); error bars = std.
  - a final panel averaged over all light levels per waveform.
  - dashed horizontal lines = free-running TEST Brain-FID (no perturbation):
    gray dashed = LIF, coloured dashed = lowBio.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

import sys
HERE = __import__("pathlib").Path(__file__).parent
pert = pd.read_csv(HERE / "perturbation_metrics.csv")
summ = pd.read_csv(HERE / "metrics_summary.csv")

# model pair: (plain model with no tau, low-rank model with swept tau)
PLAIN, LOWR = (sys.argv[1], sys.argv[2]) if len(sys.argv) > 2 else ("lif", "lowBio")
MODE = sys.argv[3] if len(sys.argv) > 3 else "freerun"   # "freerun" or "ar"
YMAX = float(sys.argv[4]) if len(sys.argv) > 4 else None  # shared y-limit across figures
RANK = int(sys.argv[5]) if len(sys.argv) > 5 else None    # fix low-rank rank (else mean over ranks)

LIF_C = "#7f7f7f"      # gray
LOW_C = "#2a9d8f"      # teal
WAVES = ["pulse", "fast_pulses", "raised_cosine"]
LEVELS = [1.3, 1.7, 2.0]
TAUS = [2, 5]

plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update({"font.size": 11, "axes.titlesize": 12, "axes.edgecolor": "#444"})

# --- test (free-running, no perturbation) Brain-FID references ---
fs = summ[(summ["mode"] == MODE) & (summ["constraint"] == "dale")]
if RANK is not None:
    fs_low = fs[fs["rank"] == RANK]
else:
    fs_low = fs
lif_test = fs[fs["model"] == PLAIN]["Brain_FID"].mean()
low_test = {t: fs_low[(fs_low["model"] == LOWR) & (fs_low["tau"] == t)]["Brain_FID"].mean()
            for t in TAUS}

base = pert[(pert["mode"] == MODE) & (pert["constraint"] == "dale")]
if RANK is not None:
    base = base[(base["model"] == PLAIN) | (base["rank"] == RANK)]
lif = base[base["model"] == PLAIN]


def agg(df, by):
    g = df.groupby(by)["Brain_FID"]
    return g.mean(), g.std().fillna(0.0)


for tau in TAUS:
    low = base[(base["model"] == LOWR) & (base["tau"] == tau)]
    low_t = low_test[tau]

    fig, axes = plt.subplots(1, 4, figsize=(19, 4.6), sharey=True)
    rank_lbl = f"rank {RANK}, " if RANK is not None else ""
    spread = "mean ± std over opto scale" + ("" if RANK is not None else " & rank")
    fig.suptitle(f"[{MODE}]  Brain-FID (↓)  —  dale-{PLAIN} vs dale-{LOWR}  "
                 f"({LOWR} {rank_lbl}τ = {tau}, {spread})", fontsize=14, y=1.02)
    w = 0.38

    # --- one panel per waveform: x = light level ---
    for ax, wave in zip(axes[:3], WAVES):
        lm, ls = agg(lif[lif["perturbation"] == wave], "light")
        cm, cs = agg(low[low["perturbation"] == wave], "light")
        x = np.arange(len(LEVELS))
        ax.bar(x - w / 2, [lm.get(l, np.nan) for l in LEVELS], w,
               yerr=[ls.get(l, 0) for l in LEVELS], capsize=3,
               color=LIF_C, edgecolor="black", linewidth=0.7, label=f"{PLAIN} (dale)")
        ax.bar(x + w / 2, [cm.get(l, np.nan) for l in LEVELS], w,
               yerr=[cs.get(l, 0) for l in LEVELS], capsize=3,
               color=LOW_C, edgecolor="black", linewidth=0.7, label=f"{LOWR} (dale)")
        ax.axhline(lif_test, ls="--", lw=1.6, color=LIF_C)
        ax.axhline(low_t, ls="--", lw=1.6, color=LOW_C)
        ax.set_title(wave)
        ax.set_xticks(x)
        ax.set_xticklabels([str(l) for l in LEVELS])
        ax.set_xlabel("light level")

    # --- summary panel: averaged over all light levels, x = waveform ---
    ax = axes[3]
    lm = lif.groupby("perturbation")["Brain_FID"].mean()
    cm = low.groupby("perturbation")["Brain_FID"].mean()
    ls_ = lif.groupby("perturbation")["Brain_FID"].std().fillna(0)
    cs_ = low.groupby("perturbation")["Brain_FID"].std().fillna(0)
    x = np.arange(len(WAVES))
    ax.bar(x - w / 2, [lm.get(v, np.nan) for v in WAVES], w,
           yerr=[ls_.get(v, 0) for v in WAVES], capsize=3,
           color=LIF_C, edgecolor="black", linewidth=0.7)
    ax.bar(x + w / 2, [cm.get(v, np.nan) for v in WAVES], w,
           yerr=[cs_.get(v, 0) for v in WAVES], capsize=3,
           color=LOW_C, edgecolor="black", linewidth=0.7)
    ax.axhline(lif_test, ls="--", lw=1.6, color=LIF_C)
    ax.axhline(low_t, ls="--", lw=1.6, color=LOW_C)
    ax.set_title("averaged over all intensities")
    ax.set_xticks(x)
    ax.set_xticklabels(WAVES, rotation=20, ha="right")

    if YMAX:
        axes[0].set_ylim(0, YMAX)          # shared across axes via sharey=True
    suffix = "_shared" if YMAX else ""
    axes[0].set_ylabel("Brain-FID")
    handles = [Patch(facecolor=LIF_C, edgecolor="black", label=f"{PLAIN} (dale) — perturbation"),
               Patch(facecolor=LOW_C, edgecolor="black", label=f"{LOWR} (dale) — perturbation"),
               Line2D([0], [0], color=LIF_C, ls="--", lw=1.6, label=f"{PLAIN} — test (no pert.)"),
               Line2D([0], [0], color=LOW_C, ls="--", lw=1.6, label=f"{LOWR} — test (no pert.)")]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, -0.08))
    fig.tight_layout()
    rank_tag = f"_rank{RANK}" if RANK is not None else ""
    out = HERE / f"barplot_brainfid_{MODE}_{PLAIN}_{LOWR}{rank_tag}_tau{tau}{suffix}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print("wrote", out)
