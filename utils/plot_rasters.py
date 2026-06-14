"""Raster-plot helpers for the perturbation-testing tutorial.

`plot_rasters` is the single master figure shared by the live training view and
the final plots; `matched_pairs` picks the example trials it shows, and
`_share_yscale` keeps the population-average panels comparable.
"""

import numpy as np
import matplotlib.pyplot as plt

from perturbation_testing import trial_matched_mse


def matched_pairs(z_gt, z_gen, feature_fun, k=2):
    """Use `trial_matched_mse`'s feature-space assignment (gen -> gt), then
    return the `k` best-matched pairs (lowest feature-space MSE). Each is
    `(gt_idx, gen_idx, mse)`. This makes the single-trial panels matched."""
    _, _, ig, idd = trial_matched_mse(z_gen, z_gt, feature_fun)   # gen ig <-> gt idd
    f_g = np.asarray(feature_fun(z_gen)); f_d = np.asarray(feature_fun(z_gt))
    mse = ((f_g[ig] - f_d[idd]) ** 2).mean(1)                     # per matched pair
    order = np.argsort(mse)[:k]
    return [(int(idd[o]), int(ig[o]), float(mse[o])) for o in order]


def _share_yscale(axes):
    """Give every trace axis the same 0..max y-range so the population-average
    firing is comparable across panels."""
    ymax = max((float(l.get_ydata().max()) for a in axes for l in a.get_lines()),
               default=1.0)
    for a in axes:
        a.set_ylim(0, ymax * 1.05 or 1.0)


def plot_rasters(rows, area_per_neuron, subset, title):
    """Master raster figure shared by the live training view and the final plots.

    `rows`: list of `(label, gt, gen, light)`, `gt`/`gen` shape (T, N), `light`
    a (T,) LED trace or None. Each raster has area-labelled y-ticks; underneath,
    the mean activity over `subset` neurons (+ the full-population mean on the
    first / trial-averaged row) and, where `light` is given, i(t) on a twin axis.
    All trace panels share one y-scale. Returns `(fig, gen, traces)` with
    `gen[r] = (im, subset_line, full_line, trace_ax)` (generated-side artists for
    live updates) and `traces` the list of trace axes to re-`_share_yscale`."""
    apn = np.asarray(area_per_neuron)
    bnd = np.concatenate([[0], np.where(apn[1:] != apn[:-1])[0] + 1, [len(apn)]])
    ticks, labels = (bnd[:-1] + bnd[1:]) / 2, apn[bnd[:-1]]
    n = len(rows)
    fig, ax = plt.subplots(2 * n, 2, figsize=(9, 3.2 * n), squeeze=False,
                           gridspec_kw={"height_ratios": [3, 1] * n},
                           constrained_layout=True)
    gen, traces = [], []
    for r, (label, gt, gn, light) in enumerate(rows):
        vmax = max(float(gt.max()), float(gn.max())) or 1.0   # shared L/R color scale
        for c, dat, side in ((0, gt, "ground truth"), (1, gn, "generated")):
            top, bot = ax[2 * r, c], ax[2 * r + 1, c]
            traces.append(bot)
            im = top.imshow(dat.T, aspect="auto", vmin=0, vmax=vmax)
            top.set_title(f"{side} — {label}", fontsize=9)
            top.tick_params(labelbottom=False)
            top.set_yticks(ticks); top.set_yticklabels(labels, fontsize=7)
            for b in bnd[1:-1]:
                top.axhline(b - 0.5, color="w", lw=0.6)
            sline, = bot.plot(dat[:, subset].mean(1), color="C0", lw=1, label="exc")
            fline, = (bot.plot(dat.mean(1), color="0.5", lw=1, ls="--", label="all")
                      if r == 0 else (None,))
            bot.set_xlim(0, dat.shape[0] - 1)
            bot.set_xlabel("time bin"); bot.set_ylabel("mean act.", fontsize=8)
            if r == 0 and c == 0:
                bot.legend(fontsize=6, loc="upper right")
            if light is not None:
                lax = bot.twinx()
                lax.plot(light, color="C1", lw=1)
                lax.set_ylabel("LED", fontsize=7, color="C1")
                lax.tick_params(axis="y", labelsize=6, labelcolor="C1")
            if c == 1:
                gen.append((im, sline, fline, bot))
    _share_yscale(traces)
    fig.suptitle(title)
    return fig, gen, traces
