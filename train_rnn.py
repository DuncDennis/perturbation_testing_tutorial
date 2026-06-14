import argparse
import os
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from data.dataloader import (load_data, get_perturbation_trials,
                             find_PV_neurons, DT)
from perturbation_testing import (evaluate, format_metrics, psth_pearson,
                                   brain_fid, trial_matched_r2, trial_matched_mse,
                                   make_population_oscillation_features,
                                   make_population_oscillation_features_torch,
                                   trial_matched_mse_loss)
from models.initialize_signed_weight_matrices import init_signed_W


class RNN(nn.Module):
    """Vanilla rate RNN and shared base for the spontaneous-activity generators.

    Dynamics: x_t = relu(x_{t-1} @ W + b + noise). No external input — the noise
    is the only drive. `sign_vector` (optional) enables Dale's law: W is init
    sign-constrained and E/I-balanced (Sourmpis 2023) and `apply_constraint`
    re-projects the signs after every optimizer step. `LIF` derives from this.
    """

    def __init__(self, n_neurons, sign_vector=None):
        super().__init__()
        if sign_vector is not None:
            W0, sign_matrix = init_signed_W(sign_vector, p0=0.5)
            assert torch.all(torch.tensor(sign_vector)[:, None] * W0 >= 0), "wrong signs"
            self.W = nn.Parameter(W0)
            self.register_buffer("sign_matrix", sign_matrix)
        else:
            self.W = nn.Parameter(torch.randn(n_neurons, n_neurons) / n_neurons ** 0.5)
            self.sign_matrix = None
        self.b = nn.Parameter(torch.zeros(n_neurons))
        # Noise is the only drive of this spontaneous generator, so it must start
        # nonzero — otherwise a ReLU rate net sits at x=relu(0)=0 with zero
        # gradient and never learns.
        self.sigma_noise = nn.Parameter(torch.ones(n_neurons) * 0.1)
        self.scale = nn.Parameter(torch.ones(1))

    @torch.no_grad()
    def apply_constraint(self):
        """Call after each optimizer step: zero the diagonal, enforce Dale's-law
        signs (if constrained), and re-pin the spectral radius. The recurrence has
        little/no leak, so without this scale control an expansive W blows the
        dynamics up to inf over a trial (the E/I matrix is strongly non-normal)."""
        self.W.data.fill_diagonal_(0.0)
        if self.sign_matrix is not None:
            sm = self.sign_matrix
            self.W.data[sm > 0] = self.W[sm > 0].clamp(min=0)
            self.W.data[sm < 0] = self.W[sm < 0].clamp(max=0)

        # Force stabilization with matrix eigenalues

        spectral_radius_max = 1.2
        eig = torch.linalg.eigvals(self.W).abs().max()
        if eig > spectral_radius_max:
            self.W.data.mul_(spectral_radius_max / eig)

    def generate(self, B, T, device=None, perturb_current=None):
        # perturb_current: (T, N) added to the pre-activation — the PV-opto hook
        # (+ve drives a unit, large -ve silences it via the ReLU floor).
        device = device or next(self.parameters()).device
        N = self.W.shape[1]
        # Warm-start from the previous call's DETACHED end state (no BPTT across
        # calls). Resample B rows (with replacement) from the stored batch so any
        # batch size works; cold-start from zeros only on the very first call.
        state = getattr(self, "_state", None)
        if state is None:
            x = torch.zeros(B, N, device=device)
        else:
            x = state[torch.randint(0, state.shape[0], (B,), device=device)]
        out = []
        for t in range(T):
            u = x @ self.W + self.b + self.sigma_noise * torch.randn_like(x)
            if perturb_current is not None:
                u = u + perturb_current[t]
            x = u.clip(min=0, max=1)
            out.append(x)
        self._state = x.detach()
        return torch.stack(out, dim=1) * self.scale


class LIF(RNN):
    """Leaky integrate-and-fire RNN with per-synapse transmission delays.

    Membrane `v` leaks (factor `al`) and integrates delayed recurrent spikes;
    spikes fire via a straight-through Bernoulli (forward = sampled 0/1, gradient
    through p = sigmoid(v)). Each i->j synapse is assigned at init to one of
    `num_delays` delay bins via the one-hot `W_d`, so the delayed weight tensor is
    `W_d[i,j,k] * W[i,j]` and each synapse reads its presynaptic spike from its
    own delay in the past.
    """

    def __init__(self, n_neurons, sign_vector=None, al=0.5, v_thr=0.1, num_delays=3):
        super().__init__(n_neurons, sign_vector)
        self.register_buffer("al", torch.tensor(float(al)))
        self.register_buffer("v_thr", torch.tensor(float(v_thr)))
        W_d = nn.functional.one_hot(
            torch.randint(0, num_delays, (n_neurons, n_neurons)), num_delays)
        self.register_buffer("W_d", W_d)            # (N, N, num_delays) one-hot
        self.temp = 1 #nn.Parameter(torch.ones(1) * 0.1)


    def generate(self, B, T, device=None, perturb_current=None):
        device = device or next(self.parameters()).device
        N, n_delays = self.W.shape[1], self.W_d.shape[-1]
        # Warm-start the spike buffer (feeds the delayed einsum) AND the membrane
        # from the previous call's DETACHED state. Resample B rows with SHARED
        # indices (so a new trial is seeded consistently across delays and v) for
        # any batch size; cold-start from zeros only on the very first call.
        state = getattr(self, "_state", None)
        if state is None:
            p0 = 0.01 * 5 # approx 5Hz
            #z_buffer = [torch.zeros(B, N, device=device) for _ in range(n_delays)]
            z_buffer = [(torch.rand(B, N, device=device) < p0).float() for _ in range(n_delays)]
            v = torch.zeros(B, N, device=device)
        else:
            idx = torch.randint(0, state[0].shape[0], (B,), device=device)
            z_buffer = [s[idx] for s in state]
            v = self._v[idx]
        W = torch.einsum("ijk,ij->ijk", self.W_d, self.W)   # delayed weight tensor
        out = []
        for t in range(T):
            u = torch.einsum("kbi,ijk->bj", torch.stack(z_buffer), W)
            u = u + self.b + self.sigma_noise * torch.randn_like(u)
            if perturb_current is not None:
                u = u + perturb_current[t]
            reset = z_buffer[-1].detach() * self.v_thr
            v = self.al * v + (1 - self.al) * u - reset
            p = torch.sigmoid(self.temp * (v - self.v_thr))
            hard = (0.5 < p).float()
            z = p + (hard - p).detach()             # straight-through spike
            z_buffer = z_buffer[1:] + [z]
            out.append(z)
        self._state = [zz.detach() for zz in z_buffer]
        self._v = v.detach()
        return torch.stack(out, dim=1)

def low_pass(z, time_last=False):
    if not time_last:
        return low_pass(z.transpose(1, 2), time_last=True).transpose(1, 2)
    return torch.avg_pool1d(z, kernel_size=5, stride=1, padding=0)

def shuffled_baselines(z_true, seed=0, feature_fun=None):
    """Notebook-2-style chance levels for the three metrics:

      - **shuffled time**:    one shared time-bin permutation applied to
        every (trial, neuron). Cross-neuron synchrony at each (permuted)
        time is preserved; PSTH temporal structure is destroyed.
      - **shuffled neuron**:  one shared permutation of the neuron axis
        applied to every (trial, time bin). Per-neuron PSTHs survive but
        the area-pop-average reads them out from the wrong neurons.

    `z_true` is binarised first so the chance level is computed on the same
    representation the model is asked to match. Returns a flat dict with
    `fid_*` and `r2_*` keys (suffix `_time` and `_neuron`).
    """
    rng = np.random.default_rng(seed)
    z_t = (z_true > 0).astype(np.float32)            # (B, T, N)
    n_trials, n_bins, n_neurons = z_t.shape

    time_perm = rng.permutation(n_bins)
    neuron_perm = rng.permutation(n_neurons)
    z_st = z_t[:, time_perm, :]
    z_sn = z_t[:, :, neuron_perm]

    psth_st, _ = psth_pearson(z_st, z_t)
    psth_sn, _ = psth_pearson(z_sn, z_t)
    fid_kw = {"feature_fun": feature_fun} if feature_fun is not None else {}
    r2_kw = {"feature_fun": feature_fun} if feature_fun is not None else {}
    return {
        "psth_shuffled_time":   psth_st,
        "psth_shuffled_neuron": psth_sn,
        "fid_shuffled_time":    brain_fid(z_st, z_t, **fid_kw),
        "r2_shuffled_time":     trial_matched_r2(z_st, z_t, **r2_kw),
        "fid_shuffled_neuron":  brain_fid(z_sn, z_t, **fid_kw),
        "r2_shuffled_neuron":   trial_matched_r2(z_sn, z_t, **r2_kw),
    }



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


def train(args):
    t0 = time.time()
    print("Loading data...")
    c_tr, z_tr, c_te, z_te, area_per_neuron, sign_per_neuron = load_data(time_last=False)
    # Inhibitory (opto-driven) units: either the "ideal" PV set read off the
    # raised-cosine perturbation, or the waveform-based fast-spiking labels.
    if args.ideal_pv_neurons:
        sign_per_neuron = (1 - find_PV_neurons()) * 2 - 1

    # This tutorial models ongoing (spontaneous) activity only: keep the c==0
    # block and binarise to spikes. Stimulus conditions are not used.
    z_tr_bin_full = (z_tr[c_tr == 0] > 0).astype(np.float32)
    z_te_bin = (z_te[c_te == 0] > 0).astype(np.float32)

    # Opto-driven (inhibitory) vs non-driven (excitatory) units. We trace and
    # score the latter: the opto clamps the driven units identically in data and
    # model, so scoring them would inflate the metrics.
    driven = sign_per_neuron == -1
    keep = ~driven
    n_neurons = z_tr_bin_full.shape[2]
    print(f"  z_train={z_tr_bin_full.shape}  z_test={z_te_bin.shape}  "
          f"areas={list(np.unique(area_per_neuron))}  load took {time.time()-t0:.1f}s")

    device = torch.device(args.device)
    sign = sign_per_neuron if args.sign_constrained else None
    if args.model == "lif":
        model = LIF(n_neurons, sign_vector=sign, num_delays=args.num_delays).to(device)
    else:
        model = RNN(n_neurons, sign_vector=sign).to(device)
    print(f"Model: {args.model}  n_neurons={n_neurons}  device={device}  "
          f"sign_constrained={args.sign_constrained}  "
          f"params={sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    train_dl = DataLoader(TensorDataset(torch.tensor(z_tr_bin_full)),
                          batch_size=args.batch_size, shuffle=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1)

    # Population-oscillation features (per-area pop average → oscillation bands →
    # z-score vs the training set). Same pipeline for the differentiable loss
    # (torch) and the eval metrics (numpy) — one source of truth.
    feature_fun = make_population_oscillation_features(
        area_per_neuron=area_per_neuron, z_train=z_tr_bin_full, dt=DT)
    feature_fun_torch = make_population_oscillation_features_torch(
        area_per_neuron=area_per_neuron, z_train=z_tr_bin_full, dt=DT)

    baselines = shuffled_baselines(z_te_bin, feature_fun=feature_fun)
    print("Test-set chance baselines (time-shuffle ≡ pedagogical reference):")
    print(f"  shuffled time   : PSTH-r={baselines['psth_shuffled_time']:+.3f}  "
          f"FID={baselines['fid_shuffled_time']:.3f}  "
          f"R²={baselines['r2_shuffled_time']:+.3f}")
    print(f"  shuffled neuron : PSTH-r={baselines['psth_shuffled_neuron']:+.3f}  "
          f"FID={baselines['fid_shuffled_neuron']:.3f}  "
          f"R²={baselines['r2_shuffled_neuron']:+.3f}")
    # Ideal reference: real train vs real test (data-vs-data ceiling — the best
    # a generator could do given trial-to-trial variability). Train is
    # subsampled to the test-trial count so the trial-matched R² is defined.
    _idx = np.random.default_rng(0).choice(z_tr_bin_full.shape[0],
                                           size=z_te_bin.shape[0], replace=False)
    m_ideal = evaluate(z_tr_bin_full[_idx], z_te_bin, feature_fun=feature_fun,
                       time_last=False)
    print(f"  ideal (train->test) : {format_metrics(m_ideal)}")

    # Live rasters: ground truth (left) vs generated (right) — trial-avg + two
    # example trials — using the same plot_rasters as the final plots. The
    # generated artists are updated in place each epoch.
    plt.ion()
    ti, tj = 0, z_te_bin.shape[0] // 2
    examples = lambda gen: [("trial-avg", z_te_bin.mean(0), gen[0], None),
                            (f"trial {ti}", z_te_bin[ti], gen[1], None),
                            (f"trial {tj}", z_te_bin[tj], gen[2], None)]
    fig, gen, traces = plot_rasters(
        examples([z_te_bin.mean(0), z_te_bin[ti], z_te_bin[tj]]),
        area_per_neuron, keep, "epoch 0")

    for epoch in range(1, args.epochs + 1):

        model.train()

        losses = {"trial_matched": [], "trial_averaged": []}
        for (z_batch,) in train_dl:
            z_batch = z_batch.to(device, non_blocking=True)   # (B, T, N)
            B, T, N = z_batch.shape

            z_gen = model.generate(B, T, device)              # free-running (B, T, N)
            # Match the trial-averaged PSTH and the single-trial population
            # statistics (trial matching over the differentiable feature space).

            ta_loss = (low_pass(z_gen).mean(0) - low_pass(z_batch).mean(0)).pow(2).mean()
            tm_loss = trial_matched_mse_loss(low_pass(z_gen), low_pass(z_batch), feature_fun_torch).mean()

            (ta_loss * args.coeff_ta + tm_loss * args.coeff_tm).backward()
            opt.step()
            opt.zero_grad()
            model.apply_constraint()   # zero diagonal, Dale's-law signs, re-pin radius

            losses["trial_averaged"].append(ta_loss.item())
            losses["trial_matched"].append(tm_loss.item())

        # Per-epoch eval: generate as many trials as the test set and score it
        # against the recorded test trials.
        z_pred = model.generate(len(z_te_bin), z_te_bin.shape[1], device=device)
        z_pred = z_pred.detach().cpu().numpy()
        metrics = evaluate(z_pred, z_te_bin, feature_fun=feature_fun, time_last=False)
        loss_str = "  ".join(f"{k}={np.mean(v):.3f}" for k, v in losses.items())
        print(f"Epoch {epoch:3d}  {loss_str}  | test {format_metrics(metrics)}")

        # Update the generated side in place (rasters + excitatory/full traces).
        for (im, sline, fline, bot), dat in zip(
                gen, [z_pred.mean(0), z_pred[ti], z_pred[tj]]):
            im.set_data(dat.T)
            sline.set_ydata(dat[:, keep].mean(1))
            if fline is not None:
                fline.set_ydata(dat.mean(1))
        _share_yscale(traces)   # one shared y-scale across all trace panels
        fig.suptitle(f"epoch {epoch}")
        fig.canvas.draw(); plt.pause(0.01)



    # Save the per-epoch live rasters: a per-run folder if --run-dir, else `figures/`.
    out_dir = args.run_dir or "figures"
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(os.path.join(out_dir, "rasters.png"), dpi=120)

    # --- Perturbation set: drive the inhibitory units like PV-ChR2 -----------
    # Recorded opto trials are gray-screen, so we drive the model with the
    # SPONTANEOUS id (c=0). One condition: the raised-cosine LED at the strongest
    # level, added as a shared current to the driven units each step -> (T, N).
    light, z_pert, meta = get_perturbation_trials(time_last=False)   # (B, T, N)
    sel = meta["stimulus_name"] == "raised_cosine"
    sel &= meta["level"] == meta["level"][sel].max()                 # strongest level
    print(f"Perturbation set: raised_cosine @ level {meta['level'][sel][0]:.1f} ({int(sel.sum())} trials)")
    z_pert, light = (z_pert[sel] > 0).astype(np.float32), light[sel]
    light_t = torch.tensor(light.mean(0), dtype=torch.float32, device=device)   # (T,)
    drive = torch.tensor(driven.astype(np.float32), device=device)
    perturb_current = args.opto_intensity * light_t[:, None] * drive[None, :]    # (T, N)

    # Score the NON-driven units only (the driven units match trivially in data
    # and model), with a feature_fun rebuilt over that subset.
    feature_fun_kept = make_population_oscillation_features(
        area_per_neuron=area_per_neuron[keep], z_train=z_tr_bin_full[:, :, keep], dt=DT)
    print(f"Eval masks out {int(driven.sum())} opto-driven units; scoring on "
          f"{int(keep.sum())}/{len(keep)} non-driven units.")

    # Final plots: in-distribution test (no perturbation) then the PV-opto set.
    # Both generate from the spontaneous id, score the non-driven units, and
    # share plot_rasters; `light=None` for the control so no LED panel is drawn.
    def final_plot(name, z_data, perturb, title, light=None):
        z_g = model.generate(len(z_data), z_data.shape[1], device=device,
                             perturb_current=perturb).detach().cpu().numpy()
        m = evaluate(z_g[:, :, keep], z_data[:, :, keep],
                     feature_fun=feature_fun_kept, time_last=False)
        print(f"{title} | {format_metrics(m)}")
        pairs = matched_pairs(z_data[:, :, keep], z_g[:, :, keep], feature_fun_kept, k=2)
        rows = [("trial-avg", z_data.mean(0), z_g.mean(0),
                 None if light is None else light.mean(0))]
        rows += [(f"gt{gj}/gen{gi}", z_data[gj], z_g[gi],
                  None if light is None else light[gj]) for gj, gi, _ in pairs]
        fig, _, _ = plot_rasters(rows, area_per_neuron, keep, f"{title} — {format_metrics(m)}")
        fig.savefig(os.path.join(out_dir, name), dpi=120)

    final_plot("test_rasters.png", z_te_bin, None, "test (no perturbation)")
    final_plot("perturbation_rasters.png", z_pert, perturb_current,
               f"perturbation test (drive {int(driven.sum())} inhib units)", light=light)
    plt.ioff(); plt.show()



if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["rnn", "lif"], default="rnn",
                   help="rnn = vanilla rate ReLU RNN; lif = leaky integrate-and-fire "
                        "(spiking, with per-synapse delays).")
    p.add_argument("--num-delays", type=int, default=3,
                   help="LIF only: number of per-synapse transmission-delay bins.")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)

    p.add_argument("--coeff_ta", type=float, default=1.0,
                   help="weight of the trial-averaged PSTH loss.")
    p.add_argument("--coeff_tm", type=float, default=0.3,
                   help="weight of the trial-matched single-trial loss.")

    p.add_argument("--opto_intensity", type=float, default=0.5)
    p.add_argument("--sign-constrained", action="store_true",
                   help="Enforce Dale's law on W (signs from the fast-spiking "
                        "waveform classifier; Sourmpis 2023 init + per-step projection).")
    p.add_argument("--ideal-pv-neurons", action="store_true",
                   help="Label inhibitory units from the raised-cosine perturbation "
                        "(ideal PV ground truth) instead of the fast-spiking waveform.")
    p.add_argument("--run-dir", default=None,
                   help="Directory to save the raster figures (default: figures/).")
    p.add_argument("--device", default="auto")
    args = p.parse_args()

    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {args.device}")

    train(args)
