#!/usr/bin/env bash
# Run all model × sign-constraint × (rank / inter-dale / tau) combinations.
# Results go to figures/<model>_{un,sign_constrained}[_interdale][_rank<R>][_tau<T>]/.
#
# Total runs:
#   RNN / LIF : 2 × 2 = 4
#   lowRNN / lowBio : 2 models × 3 constraint configs × |RANKS| × |TAUS|
#     = 2 × 3 × 4 × 4 = 96   (with the defaults below)
#   Grand total: 100
set -e

PYTHON=".venv/bin/python"
EPOCHS=20

# Swept hyperparameters for the low-rank models. tau is the (now learnable)
# membrane time constant in time-bins; the value below is its initialisation.
RANKS="1 2 3 4"
TAUS="1 2 4 8"

# ── RNN ──────────────────────────────────────────────────────────────────────
echo "=== RNN  unconstrained ==="
$PYTHON train_rnn.py --model rnn --epochs $EPOCHS

echo "=== RNN  sign-constrained ==="
$PYTHON train_rnn.py --model rnn --sign-constrained --epochs $EPOCHS

# ── LIF ──────────────────────────────────────────────────────────────────────
echo "=== LIF  unconstrained ==="
$PYTHON train_rnn.py --model lif --epochs $EPOCHS

echo "=== LIF  sign-constrained ==="
$PYTHON train_rnn.py --model lif --sign-constrained --epochs $EPOCHS

# ── lowRNN / lowBio : sweep rank × tau × sign-constraint × inter-area Dale's law ──
for MODEL in lowRNN lowBio; do
    for RANK in $RANKS; do
        for TAU in $TAUS; do
            echo "=== $MODEL  unconstrained  rank=$RANK  tau=$TAU ==="
            $PYTHON train_rnn.py --model $MODEL --epochs $EPOCHS --rank $RANK \
                --tau-init $TAU

            echo "=== $MODEL  sign-constrained  rank=$RANK  tau=$TAU ==="
            $PYTHON train_rnn.py --model $MODEL --sign-constrained --epochs $EPOCHS \
                --rank $RANK --tau-init $TAU

            echo "=== $MODEL  sign-constrained  inter-dale  rank=$RANK  tau=$TAU ==="
            $PYTHON train_rnn.py --model $MODEL --sign-constrained --inter-area-dale \
                --epochs $EPOCHS --rank $RANK --tau-init $TAU
        done
    done
done

echo ""
echo "Done. Output directories:"
echo "  figures/rnn_{un,sign_constrained}/"
echo "  figures/lif_{un,sign_constrained}/"
echo "  figures/{lowRNN,lowBio}_{unconstrained,sign_constrained,sign_constrained_interdale}_rank{1..4}_tau{1,2,4,8}/"
echo "Each contains opto_0.10/, opto_0.50/, opto_1.00/ perturbation sub-dirs."
