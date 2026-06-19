import argparse
import os
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from data.dataloader import (load_data, get_perturbation_trials,
                             find_PV_neurons, DT)
from perturbation_testing import (evaluate, format_metrics, psth_pearson,
                                   brain_fid, trial_matched_r2,
                                   make_population_oscillation_features,
                                   make_population_oscillation_features_torch,
                                   trial_matched_mse_loss)
from models.rnn_and_spiking_rnn import RNN, LIF, lowRNN, lowBio
from utils.functions import low_pass
from utils.plot_rasters import plot_rasters, _share_yscale, matched_pairs


class _Tee:
    """Mirror writes to every stream in the list (e.g. stdout + a log file)."""
    def __init__(self, *streams):
        self._streams = streams

    def write(self, data):
        for s in self._streams:
            s.write(data)

    def flush(self):
        for s in self._streams:
            s.flush()


def shuffled_baselines(z_true, seed=0, feature_fun=None):
    """Chance levels: time-shuffle destroys PSTH structure, neuron-shuffle
    reads PSTHs out from the wrong neurons. z is binarised to match the model."""
    rng = np.random.default_rng(seed)
    z_t = (z_true > 0).astype(np.float32)            # (B, T, N)
    n_trials, n_bins, n_neurons = z_t.shape

    z_st = z_t[:, rng.permutation(n_bins), :]
    z_sn = z_t[:, :, rng.permutation(n_neurons)]

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


def _plot_epoch_metrics(epoch_metrics, out_dir, baselines=None, m_ideal=None):
    """Line plot of PSTH-r / Brain-FID / trial-R² over training epochs. In
    autoregressive mode a 4th panel tracks the matched roll-out R²."""
    epochs = [m["epoch"] for m in epoch_metrics]
    specs = [
        ("psth_pearson_r",   "PSTH Pearson r ↑",  "psth_shuffled_time", "psth_pearson_r"),
        ("brain_fid",        "Brain FID ↓",        "fid_shuffled_time",  "brain_fid"),
        ("trial_matched_r2", "Trial-matched R² ↑", "r2_shuffled_time",   "trial_matched_r2"),
    ]
    if any("rollout_r2" in m for m in epoch_metrics):       # autoregressive mode
        specs.append(("rollout_r2", "Roll-out matched R² ↑", None, None))
    fig, axes = plt.subplots(1, len(specs), figsize=(4.3 * len(specs), 4),
                             constrained_layout=True, squeeze=False)
    for ax, (key, label, bl_key, ideal_key) in zip(axes[0], specs):
        ax.plot(epochs, [m.get(key, np.nan) for m in epoch_metrics], "o-",
                ms=3, lw=1.5, label="model")
        if baselines is not None and bl_key is not None:
            ax.axhline(baselines[bl_key], color="C3", ls="--", lw=1,
                       alpha=0.8, label="shuffle-time")
        if m_ideal is not None and ideal_key is not None:
            ax.axhline(m_ideal[ideal_key], color="C2", ls="--", lw=1,
                       alpha=0.8, label="ideal ceiling")
        ax.set_xlabel("epoch")
        ax.set_title(label)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
    fig.savefig(os.path.join(out_dir, "metrics_over_epochs.png"), dpi=120)
    plt.close(fig)


def _plot_perturb_metrics(perturb_results, out_dir):
    """Bar chart comparing all perturbation conditions across the three metrics."""
    conditions = [r["condition"] for r in perturb_results]
    n = len(conditions)
    specs = [
        ("psth_pearson_r",   "PSTH Pearson r ↑"),
        ("brain_fid",        "Brain FID ↓"),
        ("trial_matched_r2", "Trial-matched R² ↑"),
    ]
    colors = ["C2" if "test" in c else "C0" if c == "sham" else "C1"
              for c in conditions]
    fig, axes = plt.subplots(1, 3, figsize=(max(9, n * 1.3 + 2), 5),
                             constrained_layout=True)
    x = np.arange(n)
    for ax, (key, label) in zip(axes, specs):
        vals = [r[key] for r in perturb_results]
        ax.bar(x, vals, color=colors)
        ax.set_xticks(x)
        ax.set_xticklabels(conditions, rotation=40, ha="right", fontsize=8)
        ax.axhline(0, color="k", lw=0.6)
        ax.set_title(label)
        ax.grid(True, axis="y", alpha=0.3)
    fig.suptitle("Perturbation metrics — non-driven neurons only\n"
                 "(green=test, blue=sham, orange=opto)")
    fig.savefig(os.path.join(out_dir, "perturbation_metrics.png"), dpi=120)
    plt.close(fig)


def train(args):
    t0 = time.time()
    print("Loading data...")
    c_tr, z_tr, c_te, z_te, area_per_neuron, sign_per_neuron = load_data(time_last=False)
    if args.ideal_pv_neurons:                        # else: waveform fast-spiking labels
        sign_per_neuron = (1 - find_PV_neurons()) * 2 - 1

    # Spontaneous block (c==0) only, binarised to spikes.
    z_tr_bin_full = (z_tr[c_tr == 0] > 0).astype(np.float32)
    z_te_bin = (z_te[c_te == 0] > 0).astype(np.float32)

    # Score the non-driven (excitatory) units only: the opto clamps driven units
    # identically in data and model, so scoring them would inflate the metrics.
    driven = sign_per_neuron == -1
    keep = ~driven
    n_neurons = z_tr_bin_full.shape[2]
    print(f"  z_train={z_tr_bin_full.shape}  z_test={z_te_bin.shape}  "
          f"areas={list(np.unique(area_per_neuron))}  load took {time.time()-t0:.1f}s")

    device = torch.device(args.device)
    sign = sign_per_neuron if args.sign_constrained else None
    if args.model == "lif":
        model = LIF(n_neurons, sign_vector=sign, num_delays=args.num_delays).to(device)
    elif args.model == 'lowBio':
        model = lowBio(n_neurons, sign_vector=sign, num_delays=args.num_delays,
                       area_idx=area_per_neuron, rank=args.rank,
                       inter_area_dale=args.inter_area_dale,
                       tau_init=args.tau_init,
                       learn_tau=not args.fixed_tau).to(device)
    elif args.model == 'lowRNN':
        model = lowRNN(n_neurons, area_idx=area_per_neuron, sign_vector=sign,
                       rank=args.rank,
                       inter_area_dale=args.inter_area_dale,
                       tau_init=args.tau_init,
                       learn_tau=not args.fixed_tau).to(device)
    else:
        model = RNN(n_neurons, sign_vector=sign).to(device)
    print(f"Model: {args.model}  n_neurons={n_neurons}  device={device}  "
          f"sign_constrained={args.sign_constrained}  "
          f"params={sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    train_dl = DataLoader(TensorDataset(torch.tensor(z_tr_bin_full)),
                          batch_size=args.batch_size, shuffle=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1)

    # Same feature pipeline for the differentiable loss (torch) and eval (numpy).
    feature_fun = make_population_oscillation_features(
        area_per_neuron=area_per_neuron, z_train=z_tr_bin_full, dt=DT)
    feature_fun_torch = make_population_oscillation_features_torch(
        area_per_neuron=area_per_neuron, z_train=z_tr_bin_full, dt=DT)

    # Autoregressive (init-from-data roll-out) mode: leading-history length the
    # init needs, a clamped roll-out length, and energy-band feature fns fit on
    # R-length crops so their z-score statistics match the roll-out window.
    h_init = model.history_len
    R_ar = int(np.clip(args.rollout_len, 6, z_tr_bin_full.shape[1] - h_init))
    if args.train_mode == "autoregressive":
        z_tr_crop = z_tr_bin_full[:, :R_ar, :]
        feature_fun_ar = make_population_oscillation_features(
            area_per_neuron=area_per_neuron, z_train=z_tr_crop, dt=DT)
        feature_fun_ar_torch = make_population_oscillation_features_torch(
            area_per_neuron=area_per_neuron, z_train=z_tr_crop, dt=DT)
        if R_ar != args.rollout_len:
            print(f"  [autoregressive] rollout_len clamped to {R_ar} bins "
                  f"(trial length {z_tr_bin_full.shape[1]}, history {h_init})")

    baselines = shuffled_baselines(z_te_bin, feature_fun=feature_fun)
    print("Test-set chance baselines:")
    print(f"  shuffled time   : PSTH-r={baselines['psth_shuffled_time']:+.3f}  "
          f"FID={baselines['fid_shuffled_time']:.3f}  "
          f"R²={baselines['r2_shuffled_time']:+.3f}")
    print(f"  shuffled neuron : PSTH-r={baselines['psth_shuffled_neuron']:+.3f}  "
          f"FID={baselines['fid_shuffled_neuron']:.3f}  "
          f"R²={baselines['r2_shuffled_neuron']:+.3f}")
    # Data-vs-data ceiling: real train vs test (train subsampled to test count so
    # trial-matched R² is defined).
    _idx = np.random.default_rng(0).choice(z_tr_bin_full.shape[0],
                                           size=z_te_bin.shape[0], replace=False)
    m_ideal = evaluate(z_tr_bin_full[_idx], z_te_bin, feature_fun=feature_fun,
                       time_last=False)
    print(f"  ideal (train->test) : {format_metrics(m_ideal)}")

    # Training raster figure: generated (right) artists are updated in place each epoch
    # and saved every --plot-every epochs.
    out_dir = args.run_dir
    n_te = z_te_bin.shape[0]
    ti, tj, tk = 0, n_te // 3, 2 * n_te // 3
    examples = lambda gen: [("trial-avg", z_te_bin.mean(0), gen[0], None),
                            (f"trial {ti}", z_te_bin[ti], gen[1], None),
                            (f"trial {tj}", z_te_bin[tj], gen[2], None),
                            (f"trial {tk}", z_te_bin[tk], gen[3], None)]
    fig, gen, traces = plot_rasters(
        examples([z_te_bin.mean(0), z_te_bin[ti], z_te_bin[tj], z_te_bin[tk]]),
        area_per_neuron, keep, "epoch 0")

    is_ar = args.train_mode == "autoregressive"
    epoch_metrics = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = ({"trial_averaged": [], "rollout_mse": [], "band": []} if is_ar
                  else {"trial_matched": [], "trial_averaged": []})
        for (z_batch,) in train_dl:
            z_batch = z_batch.to(device, non_blocking=True)   # (B, T, N)
            B, T, N = z_batch.shape

            z_gen = model.generate(B, T, device)              # free-running (B, T, N)
            ta_loss = (low_pass(z_gen).mean(0) - low_pass(z_batch).mean(0)).pow(2).mean()

            if is_ar:
                # Init the state from data, roll out, and match the smoothed
                # roll-out + energy bands of the SAME trials (matched by
                # construction — no Hungarian). Several within-trial start offsets
                # give many initial conditions per batch.
                mse_acc = z_gen.new_zeros(())
                band_acc = z_gen.new_zeros(())
                for t0 in torch.randint(h_init, T - R_ar + 1, (args.n_inits,)).tolist():
                    z_pre = z_batch[:, t0 - h_init:t0, :]
                    target = z_batch[:, t0:t0 + R_ar, :]
                    state = model.init_state_from_data(z_pre)
                    z_roll = model.generate(B, R_ar, device, init_state=state)
                    mse_acc = mse_acc + (low_pass(z_roll) - low_pass(target)).pow(2).mean()
                    band_acc = band_acc + (feature_fun_ar_torch(z_roll)
                                           - feature_fun_ar_torch(target)).pow(2).mean()
                mse_loss = mse_acc / args.n_inits
                band_loss = band_acc / args.n_inits
                loss = (ta_loss * args.coeff_ta + mse_loss * args.coeff_ar_mse
                        + band_loss * args.coeff_ar_band)
            else:
                tm_loss = trial_matched_mse_loss(
                    low_pass(z_gen), low_pass(z_batch), feature_fun_torch).mean()
                loss = ta_loss * args.coeff_ta + tm_loss * args.coeff_tm

            loss.backward()
            opt.step()
            opt.zero_grad()
            model.apply_constraint()

            losses["trial_averaged"].append(ta_loss.item())
            if is_ar:
                losses["rollout_mse"].append(mse_loss.item())
                losses["band"].append(band_loss.item())
            else:
                losses["trial_matched"].append(tm_loss.item())

        # --- free-running generative eval (kept in both modes) ---
        z_pred = model.generate(len(z_te_bin), z_te_bin.shape[1], device=device)
        z_pred = z_pred.detach().cpu().numpy()
        metrics = evaluate(z_pred, z_te_bin, feature_fun=feature_fun, time_last=False)
        metrics["epoch"] = epoch
        disp = [z_pred.mean(0), z_pred[ti], z_pred[tj], z_pred[tk]]

        if is_ar:
            # Matched roll-out eval: init all test trials from data and roll out.
            z_pre_te = torch.as_tensor(z_te_bin[:, :h_init, :],
                                       dtype=torch.float32, device=device)
            state_te = model.init_state_from_data(z_pre_te)
            z_roll_te = model.generate(len(z_te_bin), R_ar, device=device,
                                       init_state=state_te).detach().cpu().numpy()
            m_ar = evaluate(z_roll_te, z_te_bin[:, h_init:h_init + R_ar, :],
                            feature_fun=feature_fun_ar, matched=True, time_last=False)
            metrics["rollout_r2"] = m_ar["trial_matched_r2"]
            metrics["rollout_psth_r"] = m_ar["psth_pearson_r"]
            metrics["rollout_fid"] = m_ar["brain_fid"]
            # Full-length matched prediction for the raster: data prefix + roll-out
            # (re-seeded from the same data init so it lines up with the data trial).
            R_plot = z_te_bin.shape[1] - h_init
            roll_full = model.generate(len(z_te_bin), R_plot, device=device,
                                       init_state=model.init_state_from_data(z_pre_te)
                                       ).detach().cpu().numpy()
            pred_full = np.concatenate([z_te_bin[:, :h_init, :], roll_full], axis=1)
            disp = [pred_full.mean(0), pred_full[ti], pred_full[tj], pred_full[tk]]

        epoch_metrics.append(metrics)
        loss_str = "  ".join(f"{k}={np.mean(v):.3f}" for k, v in losses.items())
        tau_str = (f"  tau={float(model.tau):.3f}" if hasattr(model, "tau") else "")
        ar_str = (f"  | rollout R²={metrics['rollout_r2']:+.3f}"
                  if "rollout_r2" in metrics else "")
        print(f"Epoch {epoch:3d}  {loss_str}{tau_str}  | test {format_metrics(metrics)}{ar_str}")

        for (im, sline, fline, bot), dat in zip(gen, disp):
            im.set_data(dat.T)
            sline.set_ydata(dat[:, keep].mean(1))
            if fline is not None:
                fline.set_ydata(dat.mean(1))
        _share_yscale(traces)
        fig.suptitle(f"epoch {epoch}")
        if args.plot_every > 0 and (epoch % args.plot_every == 0 or epoch == args.epochs):
            fig.savefig(os.path.join(out_dir, f"rasters_epoch_{epoch:03d}.png"), dpi=120)

    plt.close(fig)
    _plot_epoch_metrics(epoch_metrics, out_dir, baselines=baselines, m_ideal=m_ideal)

    # --- Post-training evaluation ---
    light_all, z_pert_all, meta = get_perturbation_trials(time_last=False)
    drive = torch.tensor(driven.astype(np.float32), device=device)

    feature_fun_kept = make_population_oscillation_features(
        area_per_neuron=area_per_neuron[keep], z_train=z_tr_bin_full[:, :, keep], dt=DT)
    print(f"Eval masks out {int(driven.sum())} opto-driven units; scoring on "
          f"{int(keep.sum())}/{len(keep)} non-driven units.")

    def final_plot(name, z_data, perturb, title, save_dir, light=None):
        z_g = model.generate(len(z_data), z_data.shape[1], device=device,
                             perturb_current=perturb).detach().cpu().numpy()
        m = evaluate(z_g[:, :, keep], z_data[:, :, keep],
                     feature_fun=feature_fun_kept, time_last=False)
        print(f"{title} | {format_metrics(m)}")
        pairs = matched_pairs(z_data[:, :, keep], z_g[:, :, keep], feature_fun_kept, k=3)
        rows = [("trial-avg", z_data.mean(0), z_g.mean(0),
                 None if light is None else light.mean(0))]
        rows += [(f"gt{gj}/gen{gi}", z_data[gj], z_g[gi],
                  None if light is None else light[gj]) for gj, gi, _ in pairs]
        fig, _, _ = plot_rasters(rows, area_per_neuron, keep, f"{title} — {format_metrics(m)}")
        fig.savefig(os.path.join(save_dir, name), dpi=120)
        plt.close(fig)
        return m

    # Build conditions once: sham has light_t=None; opto stores the raw light tensor
    # (not yet scaled by intensity) so we can sweep intensities without reloading.
    opto_sel = meta["kind"] == "opto"
    sham_sel = meta["kind"] == "sham"
    base_conditions = []
    if sham_sel.any():
        base_conditions.append(("sham", 0.0, sham_sel, None))
    for stim_name in sorted(np.unique(meta["stimulus_name"][opto_sel])):
        stim_sel = opto_sel & (meta["stimulus_name"] == stim_name)
        for level in sorted(np.unique(meta["level"][stim_sel])):
            cond_sel = stim_sel & (meta["level"] == level)
            light_t = torch.tensor(
                light_all[cond_sel].mean(0), dtype=torch.float32, device=device)
            base_conditions.append((stim_name, level, cond_sel, light_t))

    for opto_intensity in args.opto_intensities:
        intensity_dir = os.path.join(out_dir, f"opto_{opto_intensity:.2f}")
        os.makedirs(intensity_dir, exist_ok=True)
        print(f"\n--- Perturbation sweep: opto_intensity={opto_intensity} ---")

        m_test = final_plot("test_rasters.png", z_te_bin, None,
                            "test (no perturbation)", intensity_dir)
        perturb_results = [{"condition": "test (in-dist)", **m_test}]

        for stim_name, level, cond_sel, light_t in base_conditions:
            label = "sham" if stim_name == "sham" else f"{stim_name} l={level:.1f}"
            fname = f"perturbation_{stim_name}_l{level:.1f}.png"
            z_cond = (z_pert_all[cond_sel] > 0).astype(np.float32)
            light_arg = None if stim_name == "sham" else light_all[cond_sel]
            perturb_current = (None if light_t is None
                               else opto_intensity * light_t[:, None] * drive[None, :])
            n = int(cond_sel.sum())
            print(f"Perturb: {label} ({n} trials)")
            m = final_plot(fname, z_cond, perturb_current, label,
                           intensity_dir, light=light_arg)
            perturb_results.append({"condition": label, **m})

        _plot_perturb_metrics(perturb_results, intensity_dir)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["rnn", "lif", "lowRNN", "lowBio"], default="rnn")
    p.add_argument("--num-delays", type=int, default=3)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--coeff_ta", type=float, default=1.0)
    p.add_argument("--coeff_tm", type=float, default=0.3)
    p.add_argument("--train-mode", choices=["trialmatch", "autoregressive"],
                   default="trialmatch", dest="train_mode",
                   help="trialmatch: free-running PSTH + optimal-transport "
                        "trial-matched loss (default). autoregressive: init the "
                        "state from data and roll out, matching the smoothed "
                        "roll-out + energy bands of the SAME trials.")
    p.add_argument("--rollout-len", type=int, default=50, dest="rollout_len",
                   help="roll-out length in time-bins (autoregressive mode)")
    p.add_argument("--n-inits", type=int, default=4, dest="n_inits",
                   help="within-trial initial conditions per batch "
                        "(autoregressive mode)")
    p.add_argument("--coeff_ar_mse", type=float, default=1.0,
                   help="weight of the smoothed roll-out MSE (autoregressive mode)")
    p.add_argument("--coeff_ar_band", type=float, default=0.3,
                   help="weight of the matched energy-band MSE (autoregressive mode)")
    p.add_argument("--rank", type=int, default=2,
                   help="inter-area low-rank (lowRNN / lowBio only)")
    p.add_argument("--tau-init", type=float, default=2.0, dest="tau_init",
                   help="initial membrane time constant in time-bins "
                        "(lowRNN / lowBio only)")
    p.add_argument("--fixed-tau", action="store_true", dest="fixed_tau",
                   help="hold the time constant fixed at --tau-init instead of "
                        "learning it (lowRNN / lowBio only)")
    p.add_argument("--inter-area-dale", action="store_true", dest="inter_area_dale",
                   help="enforce Dale's law for inter-area connections (lowRNN/lowBio only)")
    p.add_argument("--opto-intensities", type=float, nargs="+",
                   default=[0.1, 0.5, 1.0], dest="opto_intensities")
    p.add_argument("--plot-every", type=int, default=5,
                   help="save a training raster every N epochs (0 = never)")
    p.add_argument("--sign-constrained", action="store_true")
    p.add_argument("--ideal-pv-neurons", action="store_true")
    p.add_argument("--run-dir", default=None)
    p.add_argument("--device", default="auto")
    args = p.parse_args()

    if args.device == "auto":
        if torch.cuda.is_available():
            args.device = "cuda"
        elif torch.backends.mps.is_available():
            args.device = "mps"
        else:
            args.device = "cpu"

    if args.run_dir is None:
        is_low = args.model in ("lowRNN", "lowBio")
        rank_suffix = f"_rank{args.rank}" if is_low else ""
        tau_suffix = (f"_tau{args.tau_init:g}{'fixed' if args.fixed_tau else ''}"
                      if is_low else "")
        dale_suffix = ("_interdale"
                       if is_low and args.sign_constrained and args.inter_area_dale
                       else "")
        mode_suffix = (f"_ar_R{args.rollout_len}"
                       if args.train_mode == "autoregressive" else "")
        args.run_dir = os.path.join(
            "figures",
            f"{args.model}_{'sign_constrained' if args.sign_constrained else 'unconstrained'}"
            f"{dale_suffix}{rank_suffix}{tau_suffix}{mode_suffix}"
        )
    os.makedirs(args.run_dir, exist_ok=True)

    _log = open(os.path.join(args.run_dir, "train.log"), "w")
    sys.stdout = _Tee(sys.__stdout__, _log)
    try:
        print(f"Using device: {args.device}")
        train(args)
    finally:
        sys.stdout = sys.__stdout__
        _log.close()
