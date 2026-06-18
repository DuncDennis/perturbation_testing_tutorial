"""Perturbation-testing evaluation metrics (used for both the in-distribution
test set and the optogenetic-perturbation set).

Convention: rasters are `(n_trials, n_bins, n_neurons)` = (B, T, N), with time
as the second axis (the RNN-natural layout). `evaluate(..., time_last=True)`
accepts the Allen/dataloader (B, N, T) layout and transposes it in.

Three metrics, all comparing model-generated rasters to held-out recorded
rasters of the same shape `(n_trials, n_bins, n_neurons)`:

  1. PSTH Pearson correlation — per-neuron Pearson r between trial-averaged
     firing rates (model vs data). Roughly the paper's `L_neuron`.
  2. Brain FID — Fréchet distance between Gaussian fits to the distribution
     of single-trial feature vectors (model vs data). Distribution-level.
  3. Trial-matched R² — R² between matched single-trial populations on the
     same feature embedding used by FID. Roughly the paper's `L_trial`.

CLAUDE.md note: "Brain FID and trial-matched R² are treated as equivalent in
this tutorial." That equivalence is made explicit by sharing a single
`feature_fun(z) -> (n_trials, d_feat)` that both metrics consume — pick the
feature once, get both numbers consistently.
"""

import numpy as np
import torch
from scipy.linalg import sqrtm
from scipy.optimize import linear_sum_assignment
from sympy.physics.mechanics import kane


# -----------------------------------------------------------------------------
# Feature functions
# -----------------------------------------------------------------------------

def time_window_features(z, n_windows=5):
    """Mean firing rate per neuron over `n_windows` equal-length time windows.

    `z`: (n_trials, n_bins, n_neurons) = (B, T, N). Returns
    (n_trials, n_neurons * n_windows). Per-neuron, raw spike-rate units.
    """
    n_trials, n_bins, n_neurons = z.shape
    edges = np.linspace(0, n_bins, n_windows + 1, dtype=int)
    feats = np.stack([z[:, edges[k]:edges[k + 1], :].mean(axis=1)
                      for k in range(n_windows)], axis=-1)        # (B, N, W)
    return feats.reshape(n_trials, -1).astype(np.float64)


def feature_temporal_average(x_btd, n_group=3):
    """Project each population channel onto log-spaced cosine/sine bands and
    return the per-band amplitude plus a DC offset. `x_btd` is `(B, T, d)`.

    Mirrors notebook-2's `feature_oscillation_frequency`. Returns
    `(B, d * (n_group + 1))` after grouping `num_f` frequencies into
    `n_group` averaged bands. The trailing DC channel preserves a per-trial
    deviation from the global mean.
    """
    x = np.asarray(x_btd, dtype=np.float64)
    B, T, d = x.shape
    x = x - x.mean(axis=(0, 1), keepdims=True)
    coeff = (f_max / f_min) ** (1.0 / (num_f - 1))
    freqs = f_min * coeff ** np.arange(num_f)                  # (F,)
    time_line = np.arange(T) * dt                              # (T,)
    cos = np.cos(np.pi * freqs[None, :] * time_line[:, None])  # (T, F)
    sin = np.sin(np.pi * freqs[None, :] * time_line[:, None])  # (T, F)
    z_cos = (x[..., None] * cos[None, :, None, :]).mean(axis=1)  # (B, d, F)
    z_sin = (x[..., None] * sin[None, :, None, :]).mean(axis=1)
    z_amp = np.sqrt(z_cos ** 2 + z_sin ** 2)
    z_amp = z_amp.reshape(B, d, n_group, num_f // n_group).mean(-1)  # (B, d, n_group)
    z_dc = x.mean(axis=1)[:, :, None]                                # (B, d, 1)
    z = np.concatenate([z_amp, z_dc], axis=-1).reshape(B, -1)
    return z

# -----------------------------------------------------------------------------
# Torch versions (differentiable; used by the trial-matched MSE loss)
# -----------------------------------------------------------------------------

def pop_averaged_with_area_torch(z_btn, area_per_neuron):
    """Differentiable torch port of `pop_averaged_with_area`.
    `z_btn`: (B, T, N) torch tensor. `area_per_neuron`: length-N numpy/list.
    Returns `(B, T, n_areas)` torch tensor with areas in stable sorted order.
    """
    areas = np.unique(area_per_neuron)
    out = torch.stack(
        [z_btn[:, :, area_per_neuron == a].mean(dim=2) for a in areas],
        dim=-1,
    )
    return out


def feature_oscillation_frequency_torch(x_btd, dt=0.01, f_min=2.0, f_max=30.0,
                                         num_f=96, n_group=3):
    """Differentiable torch port of `feature_oscillation_frequency`.
    `x_btd`: (B, T, d) torch tensor → (B, d*(n_group+1))."""
    B, T, d = x_btd.shape
    x = x_btd - x_btd.mean(dim=(0, 1), keepdim=True)
    coeff = (f_max / f_min) ** (1.0 / (num_f - 1))
    freqs = f_min * coeff ** torch.arange(
        num_f, device=x.device, dtype=x.dtype)                  # (F,)
    time_line = torch.arange(T, device=x.device, dtype=x.dtype) * dt
    arg = np.pi * freqs[None, :] * time_line[:, None]           # (T, F)
    cos = torch.cos(arg)
    sin = torch.sin(arg)
    z_cos = (x[..., None] * cos[None, :, None, :]).mean(dim=1)  # (B, d, F)
    z_sin = (x[..., None] * sin[None, :, None, :]).mean(dim=1)
    z_amp = torch.sqrt(z_cos.pow(2) + z_sin.pow(2) + 1e-12)
    z_amp = z_amp.reshape(B, d, n_group, num_f // n_group).mean(-1)  # (B,d,G)
    z_dc = x.mean(dim=1).unsqueeze(-1)                          # (B, d, 1)
    z = torch.cat([z_amp, z_dc], dim=-1).reshape(B, -1)
    return z


def make_population_oscillation_features_torch(area_per_neuron, z_train,
                                                 dt=0.01, f_min=2.0,
                                                 f_max=20.0, num_f=96,
                                                 n_group=3,
                                                 normalize=True):
    """Build a torch feature_fun closure matching `make_population_oscillation
    _features` (numpy) but with gradient flow through `z`. Captures the
    train-set μ/σ on a torch buffer so the same normalisation is applied
    everywhere. `z_train` is (B, T, N); may be numpy or torch (will be
    torched on CPU then moved at call-time)."""
    z_train_torch = torch.as_tensor(np.asarray(z_train),
                                     dtype=torch.float32)


    def raw(z_btn):
        pop = pop_averaged_with_area_torch(z_btn, area_per_neuron) # (B, T, n_areas)
        pop = (pop - pop.mean())
        return feature_oscillation_frequency_torch(
            pop, dt=dt, f_min=f_min, f_max=f_max, num_f=num_f, n_group=n_group) # (B, d_feat)

    with torch.no_grad():
        f_train = raw(z_train_torch)
        mu = f_train.mean(dim=0)
        std = f_train.std(dim=0) + 1e-6

    if not normalize:
        return raw

    def normalized(z_btn):
        f = raw(z_btn)
        return (f - mu.to(f.device)) / std.to(f.device)

    return normalized


# -----------------------------------------------------------------------------
# Trial-matched MSE training loss
# -----------------------------------------------------------------------------

def trial_matched_mse_loss(z_gen_btn, z_data_btn, feature_fun_torch):
    """Differentiable trial-matched MSE between generated and data rasters.

    Inputs are torch `(B, T, N)` binary rasters; `z_gen_btn` should carry
    gradient back to the generator (e.g. through ST-Bernoulli on logits).
    The assignment is solved on the *detached* cost matrix; gradient still
    flows through the matched cost entries themselves.

    Reference: Sourmpis et al., "Trial matching: capturing variability with
    data-constrained spiking neural networks", NeurIPS 2023 —
    https://papers.neurips.cc/paper_files/paper/2023/hash/ec702dd6e83b2113a43614685a7e2ac6-Abstract-Conference.html
    """
    f_g = feature_fun_torch(z_gen_btn)                    # (B, d) — grad
    f_d = feature_fun_torch(z_data_btn).detach()          # (B, d) — no grad
    if f_g.shape[0] != f_d.shape[0]:
        raise ValueError(
            "trial_matched_mse_loss requires equal batch sizes; got "
            f"{f_g.shape[0]} vs {f_d.shape[0]}")
    cost = ((f_g[:, None, :] - f_d[None, :, :]) ** 2).mean(dim=-1)  # (B, B)
    ind_g, ind_d = linear_sum_assignment(
        cost.detach().cpu().numpy())                      # constants
    return cost[ind_g, ind_d].mean()


def make_population_oscillation_features(area_per_neuron, z_train,**kwargs):
    """Numpy-output view of `make_population_oscillation_features_torch`.

    The eval metrics and the differentiable training loss therefore use the
    EXACT same feature pipeline — one implementation, no risk of a numpy/torch
    copy drifting apart. Returns a closure `(B, T, N) -> (B, d_feat)` numpy
    float64 (per-area pop-average → oscillation bands → train-set z-score).
    """

    ff = make_population_oscillation_features_torch(
        area_per_neuron, z_train, **kwargs)

    def numpy_features(z_btn):
        with torch.no_grad():
            f = ff(torch.as_tensor(np.asarray(z_btn), dtype=torch.float32))
        return f.cpu().numpy().astype(np.float64)

    return numpy_features


# -----------------------------------------------------------------------------
# Metrics
# -----------------------------------------------------------------------------

def psth_pearson(z_pred, z_true, eps=1e-9):
    """Per-neuron Pearson correlation between trial-averaged rates.

    Returns `(mean_r, per_neuron_r)`. Both inputs are shape
    `(n_trials, n_bins, n_neurons)` = (B, T, N). Trials need NOT be matched;
    only the trial-averaged PSTH per neuron is used."""
    psth_p = z_pred.mean(axis=0)                                  # (T, N)
    psth_t = z_true.mean(axis=0)
    pp = psth_p - psth_p.mean(axis=0, keepdims=True)              # center over time
    pt = psth_t - psth_t.mean(axis=0, keepdims=True)
    num = (pp * pt).sum(axis=0)                                   # (N,)
    den = np.sqrt((pp ** 2).sum(axis=0) * (pt ** 2).sum(axis=0)) + eps
    rs = num / den
    return float(np.nanmean(rs)), rs


def _gaussian_moments(features):
    """`features`: (n_samples, d) -> (mean (d,), cov (d, d))."""
    mu = features.mean(axis=0)
    sigma = np.cov(features, rowvar=False)
    return mu, sigma


def frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Fréchet distance between two Gaussians (FID formula)."""
    diff = mu1 - mu2
    # Stabilise the matrix sqrt with a tiny diagonal regulariser.
    covmean, _ = sqrtm(sigma1 @ sigma2 + eps * np.eye(sigma1.shape[0]),
                       disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff @ diff + np.trace(sigma1 + sigma2 - 2 * covmean))


def brain_fid(z_pred, z_true, feature_fun=time_window_features):
    """Brain-FID: Fréchet distance between Gaussian fits of trial-feature
    distributions of the model vs the data."""
    f_p = feature_fun(z_pred)
    f_t = feature_fun(z_true)
    return frechet_distance(*_gaussian_moments(f_p), *_gaussian_moments(f_t))


def trial_matched_mse(z_pred, z_true, feature_fun=time_window_features,
                      matched=False, eps=1e-9):
    """Trial-match single-trial population features and return
    `(mse, r2, ind_pred, ind_true)`.

    Same `feature_fun` as `brain_fid` so the metrics are comparable in
    feature space. `ind_pred`/`ind_true` are the trial assignment, so the
    same matching can be reused for plotting matched pairs.

    matched : bool
        If True, row k of `z_pred` corresponds to row k of `z_true`
        (encoder→decoder reconstruction): identity indices, element-wise MSE.
        If False, trials are unaligned (prior samples): a bijective optimal
        assignment via `scipy.optimize.linear_sum_assignment` on the
        per-trial squared-distance cost. Requires `n_pred == n_true`.
    """
    f_p = np.asarray(feature_fun(z_pred))           # (n_pred, d)
    f_t = np.asarray(feature_fun(z_true))           # (n_true, d)
    if f_p.shape[0] != f_t.shape[0]:
        raise ValueError("trial_matched_mse requires the same number of "
                         f"trials on both sides; got {f_p.shape[0]} vs "
                         f"{f_t.shape[0]}")
    if matched:
        ind_p = ind_t = np.arange(f_p.shape[0])
        mse = float(((f_p - f_t) ** 2).mean())
    else:
        cost = ((f_p[:, None, :] - f_t[None, :, :]) ** 2).mean(axis=-1)
        ind_p, ind_t = linear_sum_assignment(cost)
        mse = float(cost[ind_p, ind_t].mean())
    normalizer = float(((f_t - f_t.mean()) ** 2).mean()) + eps
    r2 = 1.0 - mse / normalizer
    return mse, r2, ind_p, ind_t


def trial_matched_r2(z_pred, z_true, feature_fun=time_window_features,
                     matched=False, eps=1e-9):
    """R² between trial-matched single-trial population features (thin wrapper
    over `trial_matched_mse`)."""
    return trial_matched_mse(z_pred, z_true, feature_fun, matched, eps)[1]


# -----------------------------------------------------------------------------
# Convenience: run all three at once
# -----------------------------------------------------------------------------

def evaluate(z_pred, z_true, feature_fun=time_window_features, matched=False,
             time_last=True):
    """Compute all three metrics; return a dict of scalars + per-neuron PSTH r.

    `z_pred` and `z_true` must have identical shape. The native layout is
    `(n_trials, n_bins, n_neurons)` = (B, T, N); pass `time_last=True`
    (default) when the inputs are the Allen/dataloader `(B, N, T)` layout —
    they are transposed to (B, T, N) internally. `matched` is forwarded to
    `trial_matched_r2`: `True` for encoder→decoder reconstruction (trial
    order preserved), `False` for prior samples (LSA matching)."""
    if isinstance(z_pred, torch.Tensor):
        z_pred = z_pred.detach().cpu().numpy()

    if isinstance(z_true, torch.Tensor):
        z_true = z_true.detach().cpu().numpy()

    if time_last:
        z_pred = np.swapaxes(z_pred, 1, 2)
        z_true = np.swapaxes(z_true, 1, 2)
    if z_pred.shape != z_true.shape:
        raise ValueError(f"shape mismatch: {z_pred.shape} vs {z_true.shape}")
    psth_r_mean, psth_r_per_neuron = psth_pearson(z_pred, z_true)
    fid = brain_fid(z_pred, z_true, feature_fun=feature_fun)
    r2 = trial_matched_r2(z_pred, z_true, feature_fun=feature_fun,
                          matched=matched)
    return {"psth_pearson_r": psth_r_mean,
            "psth_pearson_r_per_neuron": psth_r_per_neuron,
            "brain_fid": fid,
            "trial_matched_r2": r2}


def format_metrics(metrics):
    """One-line human-readable summary."""
    return (f"PSTH-r={metrics['psth_pearson_r']:.3f}  "
            f"Brain-FID={metrics['brain_fid']:.3f}  "
            f"trial-R²={metrics['trial_matched_r2']:.3f}")
