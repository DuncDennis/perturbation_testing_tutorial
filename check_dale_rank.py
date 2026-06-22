"""Train a PLAIN dale model (rnn / lif, sign-constrained) for N epochs, then
partition its single full weight matrix W by area and numerically check the rank
of the INTER-AREA sub-blocks.

Unlike lowRNN/lowBio, these models impose NO low-rank structure: W is a full
(N x N) matrix. The question is whether the trained inter-area sub-blocks end up
effectively low-rank anyway. We report the numerical rank and the effective rank
(singular values above 1e-2 / 1e-3 of the block's largest) per block.
"""
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

from data.dataloader import load_data, DT
from perturbation_testing import (make_population_oscillation_features_torch,
                                  trial_matched_mse_loss)
from models.rnn_and_spiking_rnn import RNN, LIF
from utils.functions import low_pass

MODEL = sys.argv[1] if len(sys.argv) > 1 else "rnn"     # "rnn" (daleRNN) / "lif" (daleLIF)
EPOCHS = int(sys.argv[2]) if len(sys.argv) > 2 else 50
torch.manual_seed(0); np.random.seed(0)
device = torch.device("cpu")

print(f"=== dale-{MODEL}  (sign-constrained, full W)  epochs={EPOCHS} ===", flush=True)
c_tr, z_tr, c_te, z_te, area_per_neuron, sign = load_data(time_last=False)
z_tr_bin = (z_tr[c_tr == 0] > 0).astype(np.float32)
n_neurons = z_tr_bin.shape[2]
area_per_neuron = np.asarray(area_per_neuron)
areas = np.unique(area_per_neuron)
aidx = {a: np.where(area_per_neuron == a)[0] for a in areas}
print(f"z_train={z_tr_bin.shape}  areas={list(areas)}", flush=True)

model = (LIF(n_neurons, sign_vector=sign, num_delays=3) if MODEL == "lif"
         else RNN(n_neurons, sign_vector=sign)).to(device)

dl = DataLoader(TensorDataset(torch.tensor(z_tr_bin)), batch_size=64, shuffle=True)
opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.1)
feat = make_population_oscillation_features_torch(
    area_per_neuron=area_per_neuron, z_train=z_tr_bin, dt=DT)

for epoch in range(1, EPOCHS + 1):
    model.train()
    ta_l = tm_l = 0.0
    for (zb,) in dl:
        zb = zb.to(device)
        B, T, N = zb.shape
        zg = model.generate(B, T, device)
        ta = (low_pass(zg).mean(0) - low_pass(zb).mean(0)).pow(2).mean()
        tm = trial_matched_mse_loss(low_pass(zg), low_pass(zb), feat).mean()
        loss = ta * 1.0 + tm * 0.3
        loss.backward(); opt.step(); opt.zero_grad()
        model.apply_constraint()
        ta_l += ta.item(); tm_l += tm.item()
    if epoch % 5 == 0 or epoch == 1:
        print(f"  epoch {epoch:3d}  ta={ta_l/len(dl):.4e}  tm={tm_l/len(dl):.4e}", flush=True)

# ---- numerical rank of inter-area sub-blocks of the full W ----
# Convention W[i, j] = presynaptic i -> postsynaptic j (x @ W). Inter-area block
# src->tgt = W[src_idx, tgt_idx].
print("\n--- numerical rank of inter-area sub-blocks of full W (SVD) ---", flush=True)
W = model.W.detach().float()
rows = []
inter_spectra = {}
all_spectra = {}                       # every block (incl. intra), for later plots
for tgt in areas:
    for src in areas:
        Wb = W[np.ix_(aidx[src], aidx[tgt])]
        sv = torch.linalg.svdvals(Wb)
        tol = sv.max() * max(Wb.shape) * torch.finfo(torch.float32).eps
        nrank = int((sv > tol).sum())
        eff2 = int((sv > 1e-2 * sv.max()).sum())
        eff3 = int((sv > 1e-3 * sv.max()).sum())
        full = min(Wb.shape)
        kind = "intra" if src == tgt else "inter"
        rows.append((f"{src}->{tgt}", kind, tuple(Wb.shape), full, nrank, eff2, eff3))
        all_spectra[f"{src}__{tgt}"] = sv.cpu().numpy()
        if kind == "inter":
            inter_spectra[f"{src}->{tgt}"] = sv.cpu().numpy()

# Save singular values of every block so any plot can be made without retraining.
np.savez(f"figure_experiments/svd_dale_{MODEL}.npz",
         areas=np.array(list(areas)), **all_spectra)

hdr = f"{'block':>14} {'kind':>6} {'shape':>11} {'full':>5} {'num_rank':>9} {'eff1e-2':>8} {'eff1e-3':>8}"
print(hdr); print("-" * len(hdr))
for name, kind, shape, full, nr, e2, e3 in rows:
    print(f"{name:>14} {kind:>6} {str(shape):>11} {full:>5} {nr:>9} {e2:>8} {e3:>8}", flush=True)

inter = [r for r in rows if r[1] == "inter"]
print(f"\nINTER-AREA blocks ({len(inter)} of them):")
print(f"  numerical rank : min={min(r[4] for r in inter)}  max={max(r[4] for r in inter)} "
      f"(full would be {min(r[3] for r in inter)}..{max(r[3] for r in inter)})")
print(f"  eff rank @1e-2 : min={min(r[5] for r in inter)}  max={max(r[5] for r in inter)}")
print(f"  eff rank @1e-3 : min={min(r[6] for r in inter)}  max={max(r[6] for r in inter)}")
print(f"  => inter-area blocks are {'FULL' if min(r[4] for r in inter)==min(r[3] for r in inter) else 'reduced'}"
      f"-rank numerically; effective (1e-2) rank is the meaningful 'used' dimensionality.")

# ---- plot inter-area singular-value spectra ----
fig, ax = plt.subplots(figsize=(8.5, 5))
for name, sv in inter_spectra.items():
    ax.semilogy(range(1, len(sv) + 1), sv / sv.max(), lw=0.8, alpha=0.55)
ax.axhline(1e-2, color="red", ls="--", lw=1.5, label="1e-2 (eff-rank threshold)")
ax.set_xlabel("singular value index"); ax.set_ylabel("σ / σ_max (log)")
ax.set_title(f"dale-{MODEL}: inter-area sub-block singular spectra "
             f"({len(inter_spectra)} blocks, {EPOCHS} epochs)\nfull W is NOT rank-constrained")
ax.legend(); ax.grid(True, alpha=0.3)
fig.tight_layout()
out = f"figure_experiments/interarea_rank_dale_{MODEL}.png"
fig.savefig(out, dpi=150)
print(f"\nwrote {out}", flush=True)
