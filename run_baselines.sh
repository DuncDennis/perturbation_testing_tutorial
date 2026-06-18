#!/usr/bin/env bash
# Run all model × sign-constraint × (rank / inter-dale) combinations.
# Results go to figures/<model>_{un,sign_constrained}[_interdale][_rank<R>]/.
#
# Total runs:
#   RNN / LIF : 2 × 2 = 4
#   lowRNN / lowBio : 2 models × (4 unconstrained + 4 sign_constrained + 4 interdale) = 24
#   Grand total: 28
set -e

PYTHON=".venv/bin/python"
EPOCHS=20

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

# ── lowRNN / lowBio : sweep rank × sign-constraint × inter-area Dale's law ──
for MODEL in lowRNN lowBio; do
    for RANK in 1 2 3 4; do
        echo "=== $MODEL  unconstrained  rank=$RANK ==="
        $PYTHON train_rnn.py --model $MODEL --epochs $EPOCHS --rank $RANK

        echo "=== $MODEL  sign-constrained  rank=$RANK ==="
        $PYTHON train_rnn.py --model $MODEL --sign-constrained --epochs $EPOCHS --rank $RANK

        echo "=== $MODEL  sign-constrained  inter-dale  rank=$RANK ==="
        $PYTHON train_rnn.py --model $MODEL --sign-constrained --inter-area-dale \
            --epochs $EPOCHS --rank $RANK
    done
done

echo ""
echo "Done. Output directories:"
echo "  figures/rnn_{un,sign_constrained}/"
echo "  figures/lif_{un,sign_constrained}/"
echo "  figures/{lowRNN,lowBio}_{unconstrained,sign_constrained,sign_constrained_interdale}_rank{1..4}/"
echo "Each contains opto_0.10/, opto_0.50/, opto_1.00/ perturbation sub-dirs."
