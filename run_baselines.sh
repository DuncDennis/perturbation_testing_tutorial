#!/usr/bin/env bash
# Run all 4 model × sign-constraint combinations and save results to
# figures/<model>_{un,sign_}constrained/.
set -e

PYTHON=".venv/bin/python"

echo "=== 1/4  RNN  unconstrained ==="
$PYTHON train_rnn.py --model rnn

echo "=== 2/4  RNN  sign-constrained ==="
$PYTHON train_rnn.py --model rnn --sign-constrained

echo "=== 3/4  LIF  unconstrained ==="
$PYTHON train_rnn.py --model lif

echo "=== 4/4  LIF  sign-constrained ==="
$PYTHON train_rnn.py --model lif --sign-constrained

echo "Done. Results in figures/rnn_unconstrained/, figures/rnn_sign_constrained/,"
echo "      figures/lif_unconstrained/, figures/lif_sign_constrained/"
