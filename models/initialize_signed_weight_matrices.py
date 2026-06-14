"""Sign-constrained recurrent weight-matrix initialization.

Port of `pub-sourmpis2023-neurips/models/rec_weight_matrix.py`
(Sourmpis et al. 2023, NeurIPS), adapted so the API takes a per-unit
`sign_vector` rather than a global excitatory fraction `p_exc`. This is
the **single big E/I matrix** case: no area structure. A per-area
block-diagonal variant may come later.

Convention: the recurrence is applied as `u = z @ W`, so `W[i, j]` is the
weight from presynaptic neuron `i` to postsynaptic neuron `j`. Dale's law
constrains the **presynaptic** sign, so each row `i` carries a constant
sign equal to `sign_vector[i]`.

Recipe (one call):
  1. random non-negative magnitudes `~ U[0, 1]`;
  2. zero out rows of units with sign 0 (outliers — no outgoing projections);
  3. **E/I balance per column**: per postsynaptic neuron `j`, downscale
     the larger of `sum_i(|exc|)`, `sum_i(|inh|)` so the two match —
     guarantees signed column sum = 0 after the next step (zero net drive
     into each postsynaptic neuron under uniform presynaptic activity);
  4. multiply by the `(N, N)` `sign_matrix` (`sign_vector` broadcast
     across columns, so each row inherits its presynaptic sign);
  5. normalise so `max |eigenvalue| = 1`.

Reference: https://github.com/EPFL-LCN/pub-sourmpis2023-neurips/blob/master/models/rec_weight_matrix.py
"""

import numpy as np
import torch


def build_sign_matrix(sign_vector):
    """`(N, N)` `sign_matrix` with entries in `{-1, 0, +1}` — the
    `sign_vector` broadcast across columns. With the `u = z @ W` convention
    `W[i, j]` is the weight from presynaptic `i` to postsynaptic `j`, so
    row `i` inherits `sign_vector[i]` (presynaptic neuron's fixed sign)."""
    sign = torch.as_tensor(np.asarray(sign_vector), dtype=torch.float32)
    n = sign.shape[0]
    return sign[:, None].expand(n, n).clone()


def _balance_cols_signed(W_unsigned, sign_matrix):
    """Per-column downscale so per-column sum of exc magnitudes equals
    per-column sum of inh magnitudes. After applying signs the signed
    column sum is exactly zero (zero net drive into each postsynaptic
    neuron at init under `u = z @ W`).
    """
    exc_pos = sign_matrix > 0
    inh_pos = sign_matrix < 0
    exc_sum = (W_unsigned * exc_pos).sum(dim=0)
    inh_sum = (W_unsigned * inh_pos).sum(dim=0)
    eps = 1e-12
    scale_inh = torch.where(inh_sum > exc_sum,
                             exc_sum / (inh_sum + eps),
                             torch.ones_like(inh_sum))
    scale_exc = torch.where(exc_sum > inh_sum,
                             inh_sum / (exc_sum + eps),
                             torch.ones_like(exc_sum))
    W = W_unsigned.clone()
    W = torch.where(exc_pos, W * scale_exc[None, :], W)
    W = torch.where(inh_pos, W * scale_inh[None, :], W)
    return W


def init_signed_W(sign_vector, p0=1.0, generator=None):
    """Build a sign-constrained `(N, N)` initial weight matrix.

    Parameters
    ----------
    sign_vector : array-like of int in `{+1, -1, 0}`, shape `(N,)`.
    p0 : float in `(0, 1]`, per-entry keep probability.
    generator : torch.Generator | None.

    Returns
    -------
    W : torch.FloatTensor `(N, N)` — sign-constrained, balanced per
        column, spectral radius 1.
    sign_matrix : torch.FloatTensor `(N, N)` with entries in `{-1, 0, +1}`.
    """
    sign_matrix = build_sign_matrix(sign_vector)
    n = sign_matrix.shape[0]
    W = torch.rand((n, n), generator=generator)
    W = W - torch.diag(torch.diag(W)) # init with zero diagonal
    if p0 < 1.0:
        keep = torch.rand((n, n), generator=generator) < p0
        W = W * keep.float()
    W = torch.where(sign_matrix == 0, torch.zeros_like(W), W)
    W = _balance_cols_signed(W, sign_matrix)
    W_signed = W * sign_matrix
    eig = torch.linalg.eigvals(W_signed)
    max_abs = eig.abs().max()
    if max_abs > 0:
        W_signed = W_signed / max_abs
    return W_signed, sign_matrix


if __name__ == "__main__":
    # Tests on the actual session's sign vector (fast-spiking classifier
    # from `data.dataloader.load_data`), keeping its size — no bootstrap
    # padding to 500. Outlier units (sign 0) are kept; the init zeros
    # their rows (no outgoing projections), so the column-sum-zero check
    # is over non-trivial postsynaptic columns only.
    #   1) max |eig| ≤ 1
    #   2) ≥ 30% of |eig| > 0.3
    #   3) per-column signed sum ≈ 0 (zero net presynaptic drive at init)
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from data.dataloader import load_data

    torch.manual_seed(0)
    _, _, _, _, _, sign_real = load_data()
    sign = sign_real.astype(np.int64)
    n = len(sign)
    n_e = int((sign > 0).sum())
    n_i = int((sign < 0).sum())
    n_o = int((sign == 0).sum())
    print(f"session: N={n}  E={n_e}  I={n_i}  outliers={n_o}")

    W, sm = init_signed_W(sign)
    # Structural check: rows must have constant sign (presynaptic).
    row_signs = torch.where(W > 0, 1, torch.where(W < 0, -1, 0))
    for i in range(n):
        s_i = int(sign[i])
        nz = row_signs[i][row_signs[i] != 0]
        if s_i == 0:
            assert nz.numel() == 0, f"row {i} (outlier) has nonzeros"
        else:
            assert (nz == s_i).all(), (
                f"row {i} (sign={s_i}) has mixed signs: "
                f"{nz.unique().tolist()}")

    eig = torch.linalg.eigvals(W).abs()
    max_eig = float(eig.max())
    frac_above_03 = float((eig > 0.3).float().mean())
    col_sum = W.sum(dim=0)
    max_abs_col_sum = float(col_sum.abs().max())
    print(f"max|eig|={max_eig:.4f}   "
          f"frac|eig|>0.3={frac_above_03:.3f}   "
          f"max|col-sum|={max_abs_col_sum:.2e}")
    assert max_eig <= 1.0 + 1e-5, f"spectral radius {max_eig} > 1"
    assert frac_above_03 >= 0.30, f"only {frac_above_03:.1%} of |eig|>0.3"
    assert max_abs_col_sum < 1e-4, (
        f"max|col-sum|={max_abs_col_sum:.4e} not ≈ 0")
    print("OK")
