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
from models.rnn_and_spiking_rnn import RNN, LIF, lowBio
from utils.functions import low_pass
from utils.plot_rasters import plot_rasters, _share_yscale, matched_pairs


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
        model = lowBio(n_neurons, sign_vector=sign, num_delays=args.num_delays, area_idx = area_per_neuron).to(device)
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

    # Live rasters; the generated (right) artists are updated in place each epoch.
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
            ta_loss = (low_pass(z_gen).mean(0) - low_pass(z_batch).mean(0)).pow(2).mean()
            tm_loss = trial_matched_mse_loss(low_pass(z_gen), low_pass(z_batch), feature_fun_torch).mean()

            (ta_loss * args.coeff_ta + tm_loss * args.coeff_tm).backward()
            opt.step()
            opt.zero_grad()
            model.apply_constraint()

            losses["trial_averaged"].append(ta_loss.item())
            losses["trial_matched"].append(tm_loss.item())

        z_pred = model.generate(len(z_te_bin), z_te_bin.shape[1], device=device)
        z_pred = z_pred.detach().cpu().numpy()
        metrics = evaluate(z_pred, z_te_bin, feature_fun=feature_fun, time_last=False)
        loss_str = "  ".join(f"{k}={np.mean(v):.3f}" for k, v in losses.items())
        print(f"Epoch {epoch:3d}  {loss_str}  | test {format_metrics(metrics)}")

        for (im, sline, fline, bot), dat in zip(
                gen, [z_pred.mean(0), z_pred[ti], z_pred[tj]]):
            im.set_data(dat.T)
            sline.set_ydata(dat[:, keep].mean(1))
            if fline is not None:
                fline.set_ydata(dat.mean(1))
        _share_yscale(traces)
        fig.suptitle(f"epoch {epoch}")
        fig.canvas.draw(); plt.pause(0.01)

    out_dir = args.run_dir or "figures"
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(os.path.join(out_dir, "rasters.png"), dpi=120)

    # Perturbation set: raised-cosine LED at the strongest level, added as a
    # shared current to the driven units each step -> (T, N).
    light, z_pert, meta = get_perturbation_trials(time_last=False)   # (B, T, N)
    sel = meta["stimulus_name"] == "raised_cosine"
    sel &= meta["level"] == meta["level"][sel].max()
    print(f"Perturbation set: raised_cosine @ level {meta['level'][sel][0]:.1f} ({int(sel.sum())} trials)")
    z_pert, light = (z_pert[sel] > 0).astype(np.float32), light[sel]
    light_t = torch.tensor(light.mean(0), dtype=torch.float32, device=device)   # (T,)
    drive = torch.tensor(driven.astype(np.float32), device=device)
    perturb_current = args.opto_intensity * light_t[:, None] * drive[None, :]    # (T, N)

    feature_fun_kept = make_population_oscillation_features(
        area_per_neuron=area_per_neuron[keep], z_train=z_tr_bin_full[:, :, keep], dt=DT)
    print(f"Eval masks out {int(driven.sum())} opto-driven units; scoring on "
          f"{int(keep.sum())}/{len(keep)} non-driven units.")

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
    p.add_argument("--model", choices=["rnn", "lif", "lowBio"], default="rnn")
    p.add_argument("--num-delays", type=int, default=3)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--coeff_ta", type=float, default=1.0)
    p.add_argument("--coeff_tm", type=float, default=0.3)
    p.add_argument("--opto_intensity", type=float, default=0.5)
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
    print(f"Using device: {args.device}")

    train(args)
