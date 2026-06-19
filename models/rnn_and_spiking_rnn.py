"""Spontaneous-activity generators for the perturbation-testing tutorial.

Two models, sharing one base:
  - `RNN`  — vanilla rate ReLU RNN (noise-driven).
  - `LIF`  — leaky integrate-and-fire spiking RNN with per-synapse delays,
             deriving from `RNN`.

Both optionally enforce Dale's law (sign-constrained, E/I-balanced `W`).
"""

import torch
from torch import nn
import numpy as np

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
        eig = torch.linalg.eigvals(self.W.cpu()).abs().max()
        if eig > spectral_radius_max:
            self.W.data.mul_(spectral_radius_max / eig)

    # Number of leading data bins needed to build an init state (rate: just the
    # last bin). Used by the autoregressive (init-from-data) training mode.
    history_len = 1

    def init_state_from_data(self, z_pre):
        """Build a roll-out init state from recent data. `z_pre` is (B, k, N)
        binary data; the rate state is initialised to the most recent bin."""
        return z_pre[:, -1, :]

    def generate(self, B, T, device=None, perturb_current=None, init_state=None):
        # perturb_current: (T, N) added to the pre-activation — the PV-opto hook
        # (+ve drives a unit, large -ve silences it via the ReLU floor).
        # init_state: (B, N) rate vector to start from (autoregressive roll-out
        # from data). When given, the warm-start buffer is left untouched.
        device = device or next(self.parameters()).device
        N = self.W.shape[1]
        # Warm-start from the previous call's DETACHED end state (no BPTT across
        # calls). Resample B rows (with replacement) from the stored batch so any
        # batch size works; cold-start from zeros only on the very first call.
        if init_state is not None:
            x = init_state
            B = x.shape[0]
        else:
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
        if init_state is None:
            self._state = x.detach()
        return torch.stack(out, dim=1) * self.scale


def _spiking_init_from_data(z_pre, n_delays):
    """Build a (z_buffer, v) roll-out init state for a spiking model from recent
    data. `z_pre` is (B, k, N) binary data with k >= n_delays; the spike buffer
    is the last `n_delays` data bins (index 0 = oldest, -1 = newest) and the
    membrane is started at v=0 (approximate — the first steps are a transient)."""
    z_buffer = [z_pre[:, -n_delays + i, :] for i in range(n_delays)]
    v = torch.zeros_like(z_pre[:, -1, :])
    return z_buffer, v


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

    @property
    def history_len(self):
        return self.W_d.shape[-1]                    # n_delays leading bins needed

    def init_state_from_data(self, z_pre):
        """(z_buffer, v) roll-out init from the last n_delays bins of `z_pre`."""
        return _spiking_init_from_data(z_pre, self.W_d.shape[-1])

    def generate(self, B, T, device=None, perturb_current=None, init_state=None):
        device = device or next(self.parameters()).device
        N, n_delays = self.W.shape[1], self.W_d.shape[-1]
        # Warm-start the spike buffer (feeds the delayed einsum) AND the membrane
        # from the previous call's DETACHED state. Resample B rows with SHARED
        # indices (so a new trial is seeded consistently across delays and v) for
        # any batch size; cold-start from zeros only on the very first call.
        # init_state=(z_buffer, v): start a roll-out from data (warm-start buffer
        # left untouched).
        if init_state is not None:
            z_buffer, v = init_state
            B = v.shape[0]
        else:
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
        if init_state is None:
            self._state = [zz.detach() for zz in z_buffer]
            self._v = v.detach()
        return torch.stack(out, dim=1)

class lowRNN(nn.Module):
    """Rate RNN with low-rank inter-area connectivity and full-rank intra-area blocks.

    Area structure is always active (area_idx is required). Sign constraints and
    inter-area Dale's law are each optional:
      - sign_vector=None          → all blocks unconstrained
      - sign_vector + inter_area_dale=False → intra-area sign-constrained, inter-area free
      - sign_vector + inter_area_dale=True  → both intra- and inter-area sign-constrained

    Each intra-area block is a full (N_a × N_a) weight matrix.
    Each inter-area block is rank-R: W = sign_src[:,None] * (M @ N.T) when
    constrained (M, N ≥ 0); W = M @ N.T when unconstrained.
    """

    def __init__(self, n_neurons, area_idx, sign_vector=None, rank=2,
                 inter_area_dale=False, tau_init=2.0, learn_tau=True):
        super().__init__()
        self.n_neurons = n_neurons
        self.rank = rank
        self.W_dict = nn.ModuleDict()

        area_idx = np.asarray(area_idx)
        sign_vector = np.asarray(sign_vector) if sign_vector is not None else None

        self.areas = np.unique(area_idx)
        self.area_indices = {
            area: torch.as_tensor(np.where(area_idx == area)[0], dtype=torch.long)
            for area in self.areas
        }

        for tgt in self.areas:
            tgt_idx = self.area_indices[tgt]
            for src in self.areas:
                src_idx = self.area_indices[src]
                key = f"{tgt}_{src}"
                if tgt == src:
                    n_tgt = len(tgt_idx)
                    if sign_vector is not None:
                        sv_tgt = sign_vector[tgt_idx.cpu().numpy()]
                        W0, sm = init_signed_W(sv_tgt, p0=0.5)
                        assert torch.all(torch.tensor(sv_tgt)[:, None] * W0 >= 0)
                    else:
                        W0 = torch.randn(n_tgt, n_tgt) / n_tgt ** 0.5
                        sm = None
                    self.W_dict[key] = nn.ParameterDict({"W": nn.Parameter(W0)})
                    self.W_dict[key].register_buffer("sign_matrix", sm)
                else:
                    constrain = sign_vector is not None and inter_area_dale
                    if constrain:
                        sign_src = torch.as_tensor(
                            sign_vector[src_idx.cpu().numpy()], dtype=torch.float32)
                        # M, N initialised non-negative; sign_src provides the row sign.
                        M0 = torch.rand(len(src_idx), rank) * 0.02
                        N0 = torch.rand(len(tgt_idx), rank) * 0.02
                    else:
                        sign_src = None
                        M0 = torch.randn(len(src_idx), rank) * 0.02
                        N0 = torch.randn(len(tgt_idx), rank) * 0.02
                    self.W_dict[key] = nn.ParameterDict({
                        "M": nn.Parameter(M0),
                        "N": nn.Parameter(N0),
                    })
                    self.W_dict[key].register_buffer("sign_src", sign_src)

        self.b = nn.Parameter(torch.zeros(n_neurons))
        self.sigma_noise = nn.Parameter(torch.ones(n_neurons) * 0.1)
        self.scale = nn.Parameter(torch.ones(1))

        # Membrane time constant (in units of time-bins), stored in log-space so
        # tau > 0 for any real value; the recurrence uses the leak / retention
        # factor al = exp(-1/tau) in (0, 1). Larger tau -> al closer to 1 -> longer
        # memory (slower dynamics). With learn_tau=True it is a trained Parameter;
        # with learn_tau=False it is a fixed buffer held at tau_init. lowBio
        # inherits this and uses `al` as its membrane leak.
        raw_log_tau = torch.log(torch.tensor(float(tau_init)))
        if learn_tau:
            self.raw_log_tau = nn.Parameter(raw_log_tau)
        else:
            self.register_buffer("raw_log_tau", raw_log_tau)

    @property
    def tau(self):
        return torch.exp(self.raw_log_tau)

    @property
    def al(self):
        """Per-step retention/leak factor in (0, 1): al = exp(-1/tau)."""
        return torch.exp(-1.0 / self.tau)

    def get_global_W(self):
        """Assemble regional blocks into a dense (N, N) matrix.

        Convention: W[i, j] = weight from presynaptic i to postsynaptic j
        (recurrence applied as x @ W).
        """
        ref = next(self.parameters())
        W = ref.new_zeros(self.n_neurons, self.n_neurons)
        for tgt in self.areas:
            tgt_idx = self.area_indices[tgt].to(ref.device)
            for src in self.areas:
                src_idx = self.area_indices[src].to(ref.device)
                block = self.W_dict[f"{tgt}_{src}"]
                if "W" in block:
                    W_block = block["W"]
                elif block.sign_src is not None:
                    W_block = block.sign_src.to(ref.device)[:, None] * (
                        block["M"] @ block["N"].T)
                else:
                    W_block = block["M"] @ block["N"].T
                W[src_idx[:, None], tgt_idx[None, :]] = W_block

        eye = torch.eye(self.n_neurons, device=W.device, dtype=W.dtype)
        return W * (1.0 - eye)

    @torch.no_grad()
    def apply_constraint(self):
        """Re-project signs and re-pin spectral radius after each optimizer step."""
        for block in self.W_dict.values():
            if "W" in block:
                block["W"].data.fill_diagonal_(0.0)
                if block.sign_matrix is not None:
                    sm = block.sign_matrix
                    block["W"].data[sm > 0] = block["W"][sm > 0].clamp(min=0)
                    block["W"].data[sm < 0] = block["W"][sm < 0].clamp(max=0)
            else:
                if block.sign_src is not None:
                    # Keep M, N ≥ 0 so sign_src exclusively controls row sign.
                    block["M"].data.clamp_(min=0)
                    block["N"].data.clamp_(min=0)

        spectral_radius_max = 1.2
        W = self.get_global_W()
        eig = torch.linalg.eigvals(W.cpu()).abs().max()
        if eig > spectral_radius_max:
            scale = float(spectral_radius_max / eig)
            sq = scale ** 0.5
            for block in self.W_dict.values():
                if "W" in block:
                    block["W"].data.mul_(scale)
                else:
                    # Symmetric scaling: (sq·M)@(sq·N).T = scale·M@N.T
                    block["M"].data.mul_(sq)
                    block["N"].data.mul_(sq)

    # Rate state: a single bin of leading data suffices to initialise a roll-out.
    history_len = 1

    def init_state_from_data(self, z_pre):
        """Build a roll-out init state from recent data. `z_pre` is (B, k, N)
        binary data; the rate state is initialised to the most recent bin."""
        return z_pre[:, -1, :]

    def generate(self, B, T, device=None, perturb_current=None, init_state=None):
        # perturb_current: (T, N) added to the pre-activation — the PV-opto hook
        # (+ve drives a unit, large -ve silences it via the ReLU floor).
        # init_state: (B, N) rate vector to start from (autoregressive roll-out
        # from data). When given, the warm-start buffer is left untouched.
        device = device or next(self.parameters()).device
        W = self.get_global_W()
        N = W.shape[1]
        # Warm-start from the previous call's DETACHED end state (no BPTT across
        # calls). Resample B rows (with replacement) from the stored batch so any
        # batch size works; cold-start from zeros only on the very first call.
        if init_state is not None:
            x = init_state
            B = x.shape[0]
        else:
            state = getattr(self, "_state", None)
            if state is None:
                x = torch.zeros(B, N, device=device)
            else:
                x = state[torch.randint(0, state.shape[0], (B,), device=device)]
        al = self.al                       # learnable leak / time-constant factor
        out = []
        for t in range(T):
            u = x @ W + self.b + self.sigma_noise * torch.randn_like(x)
            if perturb_current is not None:
                u = u + perturb_current[t]
            # Leaky update: retain a fraction `al` of the previous rate and inject
            # (1-al) of the new (clipped) drive. al->0 recovers the old memoryless
            # map x = clip(u); al->1 makes the rate change arbitrarily slowly.
            x = al * x + (1 - al) * u.clip(min=0, max=1)
            out.append(x)
        if init_state is None:
            self._state = x.detach()
        return torch.stack(out, dim=1) * self.scale

class lowBio(lowRNN):
    """Leaky integrate-and-fire RNN with per-synapse transmission delays, 
    and low-rank interregional connections.

    Membrane `v` leaks (factor `al`) and integrates delayed recurrent spikes;
    spikes fire via a straight-through Bernoulli (forward = sampled 0/1, gradient
    through p = sigmoid(v)). Each i->j synapse is assigned at init to one of
    `num_delays` delay bins via the one-hot `W_d`, so the delayed weight tensor is
    `W_d[i,j,k] * W[i,j]` and each synapse reads its presynaptic spike from its
    own delay in the past.
    """ 
    def __init__(self, n_neurons, sign_vector=None, v_thr=0.1,
                 num_delays=3, area_idx=None, rank=2, inter_area_dale=False,
                 tau_init=2.0, learn_tau=True):
        super().__init__(n_neurons, area_idx, sign_vector, rank=rank,
                         inter_area_dale=inter_area_dale, tau_init=tau_init,
                         learn_tau=learn_tau)
        # `al` (membrane leak) is now the learnable, tau-derived property from
        # lowRNN; it is no longer a fixed buffer set here.
        self.register_buffer("v_thr", torch.tensor(float(v_thr)))
        W_d = nn.functional.one_hot(
            torch.randint(0, num_delays, (n_neurons, n_neurons)), num_delays)
        self.register_buffer("W_d", W_d)            # (N, N, num_delays) one-hot
        self.temp = 1 #nn.Parameter(torch.ones(1) * 0.1)

    @property
    def history_len(self):
        return self.W_d.shape[-1]                    # n_delays leading bins needed

    def init_state_from_data(self, z_pre):
        """(z_buffer, v) roll-out init from the last n_delays bins of `z_pre`."""
        return _spiking_init_from_data(z_pre, self.W_d.shape[-1])

    def generate(self, B, T, device=None, perturb_current=None, init_state=None):
        device = device or next(self.parameters()).device
        W_global = self.get_global_W()
        N, n_delays = W_global.shape[1], self.W_d.shape[-1]
        # Warm-start the spike buffer (feeds the delayed einsum) AND the membrane
        # from the previous call's DETACHED state. Resample B rows with SHARED
        # indices (so a new trial is seeded consistently across delays and v) for
        # any batch size; cold-start from zeros only on the very first call.
        # init_state=(z_buffer, v): start a roll-out from data (warm-start buffer
        # left untouched).
        if init_state is not None:
            z_buffer, v = init_state
            B = v.shape[0]
        else:
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
        W = torch.einsum("ijk,ij->ijk", self.W_d, W_global)   # delayed weight tensor
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
        if init_state is None:
            self._state = [zz.detach() for zz in z_buffer]
            self._v = v.detach()
        return torch.stack(out, dim=1)


        
    
