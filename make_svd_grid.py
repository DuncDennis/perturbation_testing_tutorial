"""Area-to-area singular-value grid for the two dale models.

Loads the per-block singular values saved by check_dale_rank.py and draws a
(source area) x (target area) grid; each cell overlays the SVD spectrum of that
inter/intra block for daleRNN and daleLIF. Diagonal cells are intra-area.
"""
import numpy as np
import matplotlib.pyplot as plt

data = {m: np.load(f"figure_experiments/svd_dale_{m}.npz", allow_pickle=True)
        for m in ("rnn", "lif")}
areas = list(data["rnn"]["areas"])
n = len(areas)
COL = {"rnn": "#7f7f7f", "lif": "#2a9d8f"}
LBL = {"rnn": "daleRNN", "lif": "daleLIF"}

fig, axes = plt.subplots(n, n, figsize=(3.0 * n, 2.7 * n), sharex=False)
for i, src in enumerate(areas):
    for j, tgt in enumerate(areas):
        ax = axes[i, j]
        key = f"{src}__{tgt}"
        for m in ("rnn", "lif"):
            sv = data[m][key]
            ax.semilogy(range(1, len(sv) + 1), sv, color=COL[m], lw=1.3,
                        label=LBL[m], marker=".", ms=3)
        intra = src == tgt
        ax.set_title(f"{src}→{tgt}" + ("  (intra)" if intra else ""),
                     fontsize=9, color=("#222" if intra else "#444"),
                     fontweight=("bold" if intra else "normal"))
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=7)
        if j == 0:
            ax.set_ylabel("singular value", fontsize=8)
        if i == n - 1:
            ax.set_xlabel("index", fontsize=8)

handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=2, fontsize=12, frameon=False)
fig.suptitle("Area→area singular-value spectra of W  —  daleRNN vs daleLIF "
             "(50 epochs, full Dale W)", y=0.995, fontsize=14)
fig.tight_layout(rect=[0, 0, 1, 0.965])
out = "figure_experiments/svd_grid_dale_rnn_vs_lif.png"
fig.savefig(out, dpi=140, bbox_inches="tight")
print("wrote", out)
