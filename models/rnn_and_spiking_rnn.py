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

class lowRNN(nn.Module):
    """Vanilla rate RNN and shared base for the spontaneous-activity generators.

    Dynamics: x_t = relu(x_{t-1} @ W + b + noise). No external input — the noise
    is the only drive. `sign_vector` (optional) enables Dale's law: W is init
    sign-constrained and E/I-balanced (Sourmpis 2023) and `apply_constraint`
    re-projects the signs after every optimizer step. `LIF` derives from this.
    """

    def __init__(self, n_neurons, area_idx, sign_vector=None, rank=2):
        super().__init__()
        self.n_neurons = n_neurons
        self.rank = rank
        self.W_dict = nn.ModuleDict()
        if sign_vector is not None:
            if area_idx is None:
                raise ValueError("area_idx is required for low-rank regional connectivity")
            area_idx = np.asarray(area_idx)
            sign_vector = np.asarray(sign_vector)
            if len(area_idx) != n_neurons:
                raise ValueError(
                    f"area_idx length {len(area_idx)} does not match n_neurons {n_neurons}")
            if len(sign_vector) != n_neurons:
                raise ValueError(
                    f"sign_vector length {len(sign_vector)} does not match n_neurons {n_neurons}")

            # Looping through brain regions. Each block is stored as target_source
            # but later stitched into W[source, target] for the x @ W convention.
            self.areas = np.unique(area_idx)
            self.area_indices = {
                area: torch.as_tensor(np.where(area_idx == area)[0], dtype=torch.long)
                for area in self.areas
            }
            for tgt in self.areas:
                for src in self.areas:
                    # Key for intra & inter connectivity
                    key = f"{tgt}_{src}"
                    src_idx = self.area_indices[src]
                    tgt_idx = self.area_indices[tgt]
                    if tgt == src:
                        sign_vector_tgt = sign_vector[tgt_idx.cpu().numpy()]
                        W0_tgt, sign_matrix_tgt = init_signed_W(sign_vector_tgt, p0=0.5)
                        assert torch.all(torch.tensor(sign_vector_tgt)[:, None] * W0_tgt >= 0), "wrong signs"
                        # Store the parameters
                        self.W_dict[key] = nn.ParameterDict({
                            "W": nn.Parameter(W0_tgt),
                        })
                        # Register the buffer 
                        self.W_dict[key].register_buffer("sign_matrix", sign_matrix_tgt)
                    else:
                        # Initialize low-rank connectivity 
                        self.W_dict[key] = nn.ParameterDict({
                            "M": nn.Parameter(torch.randn(len(src_idx), rank) * 0.02),
                            "N": nn.Parameter(torch.randn(len(tgt_idx), rank) * 0.02),
                        })

        else:
            self.W = nn.Parameter(torch.randn(n_neurons, n_neurons) / n_neurons ** 0.5)
            self.sign_matrix = None
            
        self.b = nn.Parameter(torch.zeros(n_neurons))
        # Noise is the only drive of this spontaneous generator, so it must start
        # nonzero — otherwise a ReLU rate net sits at x=relu(0)=0 with zero
        # gradient and never learns.
        self.sigma_noise = nn.Parameter(torch.ones(n_neurons) * 0.1)
        self.scale = nn.Parameter(torch.ones(1))

    def get_global_W(self):
        """Assemble the regional block ModuleDict into a dense (N, N) matrix.

        The recurrence uses `x @ W`, so rows are source/presynaptic neurons and
        columns are target/postsynaptic neurons. Intra-area blocks are stored
        directly as `W`; inter-area blocks are low-rank factors `M @ N.T`.
        """
        if len(self.W_dict) == 0:
            return self.W

        ref = next(self.parameters())
        W = ref.new_zeros(self.n_neurons, self.n_neurons)
        for tgt in self.areas:
            tgt_idx = self.area_indices[tgt].to(ref.device)
            for src in self.areas:
                src_idx = self.area_indices[src].to(ref.device)
                block = self.W_dict[f"{tgt}_{src}"]
                if "W" in block:
                    W_block = block["W"]
                else:
                    W_block = block["M"] @ block["N"].T
                W[src_idx[:, None], tgt_idx[None, :]] = W_block

        eye = torch.eye(self.n_neurons, device=W.device, dtype=W.dtype)
        return W * (1.0 - eye)

    @torch.no_grad()
    def apply_constraint(self):
        """Call after each optimizer step: zero the diagonal, enforce Dale's-law
        signs (if constrained), and re-pin the spectral radius. The recurrence has
        little/no leak, so without this scale control an expansive W blows the
        dynamics up to inf over a trial (the E/I matrix is strongly non-normal)."""
        if len(self.W_dict) == 0:
            self.W.data.fill_diagonal_(0.0)
            if self.sign_matrix is not None:
                sm = self.sign_matrix
                self.W.data[sm > 0] = self.W[sm > 0].clamp(min=0)
                self.W.data[sm < 0] = self.W[sm < 0].clamp(max=0)
        else:
            for block in self.W_dict.values():
                if "W" not in block:
                    continue
                block["W"].data.fill_diagonal_(0.0)
                sm = block.sign_matrix
                block["W"].data[sm > 0] = block["W"][sm > 0].clamp(min=0)
                block["W"].data[sm < 0] = block["W"][sm < 0].clamp(max=0)

        # Force stabilization with matrix eigenalues

        spectral_radius_max = 1.2
        W = self.get_global_W()
        eig = torch.linalg.eigvals(W.cpu()).abs().max()
        if eig > spectral_radius_max:
            scale = float(spectral_radius_max / eig)
            if len(self.W_dict) == 0:
                self.W.data.mul_(scale)
            else:
                for block in self.W_dict.values():
                    if "W" in block:
                        block["W"].data.mul_(scale)
                    else:
                        block["M"].data.mul_(scale)

    def generate(self, B, T, device=None, perturb_current=None):
        # perturb_current: (T, N) added to the pre-activation — the PV-opto hook
        # (+ve drives a unit, large -ve silences it via the ReLU floor).
        device = device or next(self.parameters()).device
        W = self.get_global_W()
        N = W.shape[1]
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
            u = x @ W + self.b + self.sigma_noise * torch.randn_like(x)
            if perturb_current is not None:
                u = u + perturb_current[t]
            x = u.clip(min=0, max=1)
            out.append(x)
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
    def __init__(self, n_neurons, sign_vector=None, al=0.5, v_thr=0.1,
                 num_delays=3, area_idx=None, rank=2):
        super().__init__(n_neurons, area_idx, sign_vector, rank=rank)
        self.register_buffer("al", torch.tensor(float(al)))
        self.register_buffer("v_thr", torch.tensor(float(v_thr)))
        W_d = nn.functional.one_hot(
            torch.randint(0, num_delays, (n_neurons, n_neurons)), num_delays)
        self.register_buffer("W_d", W_d)            # (N, N, num_delays) one-hot
        self.temp = 1 #nn.Parameter(torch.ones(1) * 0.1)
        
    def generate(self, B, T, device=None, perturb_current=None):
        device = device or next(self.parameters()).device
        W_global = self.get_global_W()
        N, n_delays = W_global.shape[1], self.W_d.shape[-1]
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
        self._state = [zz.detach() for zz in z_buffer]
        self._v = v.detach()
        return torch.stack(out, dim=1)


        
    
