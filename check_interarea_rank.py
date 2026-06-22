"""Train a low-rank model (lowBio / lowRNN) for N epochs, then numerically
verify the rank of the INTER-AREA blocks of the recurrent weight matrix W.

Each inter-area block is parameterised as W_block = (sign) * (M @ N^T) with
M:(n_src x R), N:(n_tgt x R), so by construction rank(W_block) <= R. Here we
check the *numerical* rank (SVD) of the trained blocks to confirm it equals the
imposed R (and is not, e.g., collapsed lower), and contrast with the full-rank
intra-area blocks.
"""
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

from data.dataloader import load_data, DT
from perturbation_testing import (make_population_oscillation_features_torch,
                                  trial_matched_mse_loss)
from models.rnn_and_spiking_rnn import lowRNN, lowBio
from utils.functions import low_pass

MODEL = sys.argv[1] if len(sys.argv) > 1 else "lowBio"
RANK = int(sys.argv[2]) if len(sys.argv) > 2 else 3
EPOCHS = int(sys.argv[3]) if len(sys.argv) > 3 else 50
torch.manual_seed(0); np.random.seed(0)
device = torch.device("cpu")

print(f"=== {MODEL}  imposed inter-area rank R={RANK}  epochs={EPOCHS} ===", flush=True)
c_tr, z_tr, c_te, z_te, area_per_neuron, sign = load_data(time_last=False)
z_tr_bin = (z_tr[c_tr == 0] > 0).astype(np.float32)
n_neurons = z_tr_bin.shape[2]
print(f"z_train={z_tr_bin.shape}  areas={list(np.unique(area_per_neuron))}", flush=True)

common = dict(area_idx=area_per_neuron, sign_vector=sign, rank=RANK,
              inter_area_dale=False, tau_init=2.0, learn_tau=True)
model = (lowBio(n_neurons, num_delays=3, **common) if MODEL == "lowBio"
         else lowRNN(n_neurons, **common)).to(device)

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
        print(f"  epoch {epoch:3d}  ta={ta_l/len(dl):.4e}  tm={tm_l/len(dl):.4e}  "
              f"tau={float(model.tau):.3f}", flush=True)

# ---- numerical rank analysis of inter-area blocks ----
print("\n--- numerical rank of W blocks (SVD) ---", flush=True)
areas = model.areas
rows = []
inter_spectra = {}
with torch.no_grad():
    for tgt in areas:
        for src in areas:
            blk = model.W_dict[f"{tgt}_{src}"]
            if "W" in blk:                       # intra-area, full-rank
                Wb = blk["W"].detach()
                kind = "intra"
            elif blk.sign_src is not None:
                Wb = blk.sign_src[:, None] * (blk["M"] @ blk["N"].T)
                kind = "inter"
            else:
                Wb = blk["M"] @ blk["N"].T
                kind = "inter"
            sv = torch.linalg.svdvals(Wb.float())
            tol = sv.max() * max(Wb.shape) * torch.finfo(torch.float32).eps
            nrank = int((sv > tol).sum())
            eff = int((sv > 1e-3 * sv.max()).sum())   # effective rank (1e-3 rel.)
            rows.append((f"{src}->{tgt}", kind, tuple(Wb.shape), nrank, eff,
                         float(sv.max())))
            if kind == "inter":
                inter_spectra[f"{src}->{tgt}"] = sv.cpu().numpy()

hdr = f"{'block':>14} {'kind':>6} {'shape':>12} {'num_rank':>9} {'eff_rank':>9} {'sv_max':>10}"
print(hdr); print("-" * len(hdr))
for name, kind, shape, nr, eff, smax in rows:
    print(f"{name:>14} {kind:>6} {str(shape):>12} {nr:>9} {eff:>9} {smax:>10.3e}", flush=True)

inter = [r for r in rows if r[1] == "inter"]
intra = [r for r in rows if r[1] == "intra"]
print(f"\nimposed R = {RANK}")
print(f"inter-area blocks: num_rank in "
      f"[{min(r[3] for r in inter)}, {max(r[3] for r in inter)}], "
      f"eff_rank(1e-3) in [{min(r[4] for r in inter)}, {max(r[4] for r in inter)}]")
print(f"intra-area blocks: num_rank in "
      f"[{min(r[3] for r in intra)}, {max(r[3] for r in intra)}] "
      f"(full = area size)")

# global W rank for reference
with torch.no_grad():
    W = model.get_global_W()
    svW = torch.linalg.svdvals(W.float())
    tolW = svW.max() * max(W.shape) * torch.finfo(torch.float32).eps
print(f"\nglobal W: shape={tuple(W.shape)}  num_rank={int((svW>tolW).sum())}")

# ---- plot inter-area singular-value spectra ----
fig, ax = plt.subplots(figsize=(8, 5))
for name, sv in inter_spectra.items():
    ax.semilogy(range(1, len(sv) + 1), sv / sv.max(), marker=".", lw=0.6, alpha=0.6)
ax.axvline(RANK + 0.5, color="red", ls="--", lw=2, label=f"imposed R = {RANK}")
ax.set_xlabel("singular value index"); ax.set_ylabel("σ / σ_max (log)")
ax.set_xlim(0.5, RANK + 4.5)
ax.set_title(f"{MODEL}: inter-area block singular spectra "
             f"({len(inter_spectra)} blocks, {EPOCHS} epochs)")
ax.legend(); ax.grid(True, alpha=0.3)
fig.tight_layout()
out = f"figure_experiments/interarea_rank_{MODEL}_R{RANK}.png"
fig.savefig(out, dpi=150)
print(f"\nwrote {out}", flush=True)
