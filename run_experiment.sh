#!/usr/bin/env bash
# ==============================================================================
# Full experiment grid for the perturbation-testing tutorial.
#
# Every model variant is trained TWICE: once free-running (trial-matching
# objective) and once autoregressively (init-from-data roll-out objective).
# Each trained model is then evaluated under the optogenetic perturbation at
# three light intensities (0.01, 0.1, 1.0).
#
# Grid
#   Baselines : {rnn, lif} x {unconstrained, Dale (sign-constrained)}      = 4
#   Low-rank  : {lowRNN, lowBio} x {unconstrained, intra-Dale}
#               x rank {1,2,3} x tau-init {2,5}                            = 24
#   Per training mode: 28 runs ; two modes (free-run + AR): 56 runs total.
#
# Choices (dt = 10 ms, trial = 200 bins = 2.0 s):
#   - tau-init {2,5} bins = {20, 50} ms : a membrane-typical and a slower
#     integration timescale (tau is the *initial* value; it is learned).
#     Only the low-rank models have a time constant.
#   - roll-out = 100 bins = 1.0 s (half the trial): long enough to resolve the
#     2 Hz band (~2 cycles) for the energy-band loss, while leaving 100 bins of
#     within-trial offset range for the sampled initial conditions.
#
# Usage
#   ./run_experiment.sh [OUTPUT_DIR]          # default: figure_experiments
#   EPOCHS=30 ./run_experiment.sh out/        # override epochs
#   DRYRUN=1 ./run_experiment.sh              # print the commands, run nothing
#   DEVICE=cuda ./run_experiment.sh           # force a device (default: auto)
#
# Output layout
#   <OUTPUT_DIR>/<freerun|ar>/<model>_<constraint>[_rank<R>_tau<T>]/
#       train.log, metrics_over_epochs.png, rasters_epoch_*.png,
#       opto_0.01/, opto_0.10/, opto_1.00/  (perturbation sub-dirs)
# ==============================================================================
set -uo pipefail

OUT="${1:-figure_experiments}"
PYTHON="${PYTHON:-.venv/bin/python}"
EPOCHS="${EPOCHS:-20}"
DEVICE="${DEVICE:-auto}"

OPTO="0.01 0.1 1.0"     # opto light intensities swept at evaluation (per run)
RANKS="1 2 3"           # inter-area low-rank for lowRNN / lowBio
TAUS="2 5"              # initial membrane time constant (bins): 20 ms, 50 ms
ROLLOUT=100             # autoregressive roll-out length (bins) = 1.0 s

count_words () { echo $#; }
N_RANKS=$(count_words $RANKS)
N_TAUS=$(count_words $TAUS)
TOTAL=$(( 2 * (4 + 2 * 2 * N_RANKS * N_TAUS) ))

COUNT=0
FAILED=()

# run <run_subdir> <train_rnn.py args...>
run () {
    local subdir="$1"; shift
    COUNT=$((COUNT + 1))
    echo ""
    echo "[$COUNT/$TOTAL] $subdir"
    if [ "${DRYRUN:-0}" = "1" ]; then
        echo "    $PYTHON train_rnn.py --epochs $EPOCHS --device $DEVICE" \
             "--opto-intensities $OPTO --run-dir $OUT/$subdir $*"
        return
    fi
    if ! $PYTHON train_rnn.py --epochs "$EPOCHS" --device "$DEVICE" \
            --opto-intensities $OPTO --run-dir "$OUT/$subdir" "$@"; then
        echo "  !! FAILED: $subdir"
        FAILED+=("$subdir")
    fi
}

echo "Output dir : $OUT"
echo "Epochs     : $EPOCHS   Device: $DEVICE   Total runs: $TOTAL"
echo "Opto       : $OPTO     Ranks: $RANKS   Taus: $TAUS   Roll-out: $ROLLOUT bins"

for MODE in trialmatch autoregressive; do
    if [ "$MODE" = "autoregressive" ]; then
        TAG="ar"
        MODEARGS="--train-mode autoregressive --rollout-len $ROLLOUT"
    else
        TAG="freerun"
        MODEARGS="--train-mode trialmatch"
    fi
    echo ""
    echo "############################################################"
    echo "## Training mode: $MODE  ($TAG)"
    echo "############################################################"

    # ── Baselines: RNN, LIF × {unconstrained, Dale} ──────────────────────────
    for M in rnn lif; do
        run "$TAG/${M}_unconstrained" --model "$M" $MODEARGS
        run "$TAG/${M}_dale"          --model "$M" --sign-constrained $MODEARGS
    done

    # ── Low-rank: lowRNN, lowBio × {unconstrained, intra-Dale} × rank × tau ──
    # "intra-Dale" = --sign-constrained WITHOUT --inter-area-dale (inter-area
    # projections stay unconstrained).
    for M in lowRNN lowBio; do
        for R in $RANKS; do
            for TAU in $TAUS; do
                run "$TAG/${M}_unconstrained_rank${R}_tau${TAU}" \
                    --model "$M" --rank "$R" --tau-init "$TAU" $MODEARGS
                run "$TAG/${M}_dale_rank${R}_tau${TAU}" \
                    --model "$M" --sign-constrained --rank "$R" --tau-init "$TAU" $MODEARGS
            done
        done
    done
done

echo ""
echo "============================================================"
if [ "${#FAILED[@]}" -eq 0 ]; then
    echo "Done. $COUNT/$TOTAL runs completed; results under $OUT/{freerun,ar}/."
else
    echo "Done with ${#FAILED[@]} failure(s):"
    printf '  - %s\n' "${FAILED[@]}"
fi
