# Perturbation testing tutorial

## Context

Minimal Python tutorial for perturbation testing of neural-activity generators
on a single Allen Neuropixels session with optogenetic stimulation of PV
inhibitory neurons. Companion to Bellec et al., eLife 106827
(https://elifesciences.org/articles/106827).

Session 829720705 (Pvalb-Cre × Ai32, *functional_connectivity*) is hard-coded;
see `SESSION_SCAN.md` for the rationale and fallbacks.

```bash
uv sync
```

## Data loading

`data/dataloader.py` builds the train / test / perturbation sets from the
drifting-gratings stimulus. Trials are stratified 70 / 30 per condition;
optogenetic trials form a separate held-out set.

```bash
uv run python data/dataloader.py
```

![dataloader summary](figures/dataloader_summary.png)

Top: example single-trial rasters. Bottom: per-area confusion matrices from
a multinomial logistic regression on the first ~600 ms of each trial.

![dataloader perturbation](figures/dataloader_perturbation.png)

Sham + the strongest level of each opto waveform (`pulse`, `fast_pulses`,
`raised_cosine`). Top row: trial-averaged firing rate. Bottom row: light
intensity `i(t)` for one example trial.

## Generative model training

`train_generative_model.py` trains the LFADS-style LSTM VAE in
`models/lstm_generator.py` and periodically logs PSTH-r, Brain FID, and
trial-matched R² on held-out trials.

```bash
uv run python train_generative_model.py --epochs 100 --eval-every 5
```

![training curves](figures/training_curves.png)

Loss, Brain FID, trial-matched R², and PSTH Pearson r over training, for
both encoder reconstruction (train) and prior samples vs held-out test.

## Perturbation testing

Not yet implemented. `perturbation_testing.py` defines the three metrics
(PSTH Pearson r, Brain FID, trial-matched R²) used during training; the
matched-perturbation evaluation on the PV-opto set, including silencing of
the model's inhibitory units, is the next step.
# perturbation_testing_tutorial
