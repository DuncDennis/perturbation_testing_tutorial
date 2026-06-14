"""Tutorial dataloader for the perturbation-testing tutorial.

Loads spike rasters from a single Allen Visual Coding Neuropixels session
(see CLAUDE.md / SESSION_SCAN.md for session 829720705) and splits them into
three sets:

    - in-distribution train         (70% of each non-perturbation condition)
    - in-distribution test          (30% of each non-perturbation condition)
    - perturbation test             (all optogenetic trials, c == -1)

Conditions are integer-coded:

    c ==  0  -> spontaneous           (gray-screen 2 s windows from the
                                       `spontaneous` stimulus block)
    c >=  1  -> drifting-gratings     (`stimulus_condition_id`s remapped to
                                       contiguous positive integers; the
                                       null/blank-sweep DG trial gets one of
                                       these positive ids)
    c == -1  -> optogenetic perturbation (2 s starting at each opto pulse onset)

All trials have the same duration (default 2 s) so the raster tensor is
rectangular: shape (n_trials, n_neurons, n_bins).
"""

import os
import sys
import numpy as np

# Make the project root importable so `data.allen_to_tensor` and
# `utils.functions` resolve when running this file as a script.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from data.allen_to_tensor import AllenToTensor


SESSION_ID = 829720705
TRIAL_DURATION_S = 2.0           # train / test trial duration (DG + spontaneous)
DT = 0.01                        # 10 ms bins
TRAIN_FRAC = 0.7
DEFAULT_DRIFTING_CONDITIONS = "ori"   # "ori" | "contrast" | "speed" | "all"
# Perturbation-set trial layout (separate from train/test).
PERTURB_DUR_S = 1.5              # perturbation trials are 1.5 s long
PERTURB_PRE_S = 0.5              # LED onset placed at t = 0.5 s into the trial
# Sham buffer is the minimum LED-free distance required around each LED edge
# for a 1.5-s sham segment to be drawn from the in-block gaps. Decoupled from
# `perturb_pre_s` since with `pre_s = 0.5` the median ISI (~1.93 s) leaves
# zero feasible sham windows.
SHAM_BUFFER_S = 0.1

# Visual cortex areas covered by the Neuropixels probes in the FC sessions.
VISUAL_AREAS = ["VISp", "VISrl", "VISl", "VISal", "VISpm", "VISam"]

# Allen-SDK caches live under data/cache/ (resolved absolutely so the script
# can be run from any working directory).
DEFAULT_CACHE_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")


def _bin_trials(session, unit_ids, trial_starts, n_bins, dt):
    """Bin spike times into a (n_trials, n_units, n_bins) int8 raster.

    Each trial spans [t0, t0 + n_bins*dt). Counts (not just 0/1) are returned;
    callers that need binary spikes can do `(z > 0).astype(...)`.
    """
    n_trials = len(trial_starts)
    n_units = len(unit_ids)
    duration = n_bins * dt
    raster = np.zeros((n_trials, n_units, n_bins), dtype=np.int8)
    for u_idx, unit_id in enumerate(unit_ids):
        st = np.asarray(session.spike_times[unit_id])
        for t_idx, t0 in enumerate(trial_starts):
            i0 = np.searchsorted(st, t0)
            i1 = np.searchsorted(st, t0 + duration)
            if i1 > i0:
                counts, _ = np.histogram(st[i0:i1] - t0, bins=n_bins,
                                         range=(0.0, duration))
                raster[t_idx, u_idx] = counts.astype(np.int8)
    return raster


def _condition_codes(dg_table, drifting_conditions):
    """Map drifting-gratings trials to contiguous positive integer condition ids.

    `drifting_conditions` selects which stimulus parameter(s) define a condition:
        "ori"      -> orientation only
        "contrast" -> contrast only
        "speed"    -> temporal_frequency only
        "all"      -> the full Allen `stimulus_condition_id`

    Null/blank-sweep trials end up in their own group (sorted last).
    """
    column = {"ori": "orientation", "contrast": "contrast",
              "speed": "temporal_frequency", "all": "stimulus_condition_id"}
    if drifting_conditions not in column:
        raise ValueError(f"drifting_conditions must be one of {list(column)}")
    raw = dg_table[column[drifting_conditions]].to_numpy()
    keys = np.array([str(v) for v in raw])

    def sort_key(k):
        try:
            return (0, float(k))
        except ValueError:
            return (1, k)  # non-numeric (e.g. "null", "nan") goes last
    unique_keys = sorted(np.unique(keys), key=sort_key)
    remap = {k: i + 1 for i, k in enumerate(unique_keys)}
    return np.array([remap[k] for k in keys], dtype=int)


def _stratified_split(c, z, train_frac, rng):
    """Per-condition stratified split into (c_train, z_train, c_test, z_test)."""
    train_idx, test_idx = [], []
    for cond in np.unique(c):
        idx = np.where(c == cond)[0]
        rng.shuffle(idx)
        n_train = int(round(train_frac * len(idx)))
        train_idx.append(idx[:n_train])
        test_idx.append(idx[n_train:])
    train_idx = np.concatenate(train_idx) if train_idx else np.array([], dtype=int)
    test_idx = np.concatenate(test_idx) if test_idx else np.array([], dtype=int)
    return c[train_idx], z[train_idx], c[test_idx], z[test_idx]


def _cache_path(cache_root, kind, **kwargs):
    """Build a stable .npz path under `cache_root/dataloader/` from a flat
    dict of args. Areas are joined with `-` so the filename keys cleanly on
    every parameter that can change the binned tensor."""
    parts = [kind]
    for k, v in kwargs.items():
        if isinstance(v, (list, tuple)):
            v = "-".join(str(x) for x in v)
        parts.append(f"{k}={v}")
    name = "_".join(parts).replace(" ", "") + ".npz"
    folder = os.path.join(cache_root, "dataloader")
    os.makedirs(folder, exist_ok=True)
    return os.path.join(folder, name)


def _area_sort_units(session, unit_ids, areas):
    """Group `unit_ids` by area in the canonical `areas` order (stable sort).
    Returns `(sorted_unit_ids, area_per_neuron)`. Shared by `load_data` and
    `get_perturbation_trials` so both produce the SAME neuron ordering."""
    unit_areas = np.asarray(
        session.units.loc[list(unit_ids), "ecephys_structure_acronym"].tolist(),
        dtype="U10")
    rank = {a: i for i, a in enumerate(areas)}
    order = np.argsort([rank.get(a, len(areas)) for a in unit_areas], kind="stable")
    return unit_ids[order], unit_areas[order]


def load_data(session_id=SESSION_ID, dt=DT, trial_duration_s=TRIAL_DURATION_S,
              areas=VISUAL_AREAS, train_frac=TRAIN_FRAC, seed=0,
              cache_root=DEFAULT_CACHE_ROOT,
              drifting_conditions=DEFAULT_DRIFTING_CONDITIONS,
              use_cache=True, time_last=True):
    """Returns ``c_train, z_train, c_test, z_test, area_per_neuron, sign_per_neuron``.

    Rasters are `(n_trials, n_neurons, n_bins)` = (B, N, T) when
    `time_last=True` (default; the Allen-native layout). Pass
    `time_last=False` to get `(B, T, N)` (time second) — the RNN-natural
    layout used by `train_rnn.py` and `perturbation_testing.py`. The cache
    always stores the canonical (B, N, T); the transpose is applied on return.

    Builds the in-distribution train/test sets (drifting-gratings +
    spontaneous, 70/30 stratified split per condition) over the union of
    `areas` in a single binning pass. Neurons are sorted so they are grouped
    by area in the canonical order specified by `areas`, and
    `area_per_neuron` (shape `(n_neurons,)`, dtype str) records each
    neuron's area for downstream slicing/plotting.

    The optogenetic perturbation set is *not* returned here — see
    `get_perturbation_trials`.

    `drifting_conditions` selects the condition id definition: "ori"
    (default), "contrast", "speed", or "all". Spontaneous trials are
    always `c == 0`.

    Results are cached as a `.npz` under `<cache_root>/dataloader/` keyed on
    every parameter that can affect the output. Pass `use_cache=False` to
    bypass.
    """
    n_bins = int(round(trial_duration_s / dt))
    os.makedirs(cache_root, exist_ok=True)
    cache_file = _cache_path(cache_root, "load_data",
                              sess=session_id, dt=dt, dur=trial_duration_s,
                              areas=areas, frac=train_frac, seed=seed,
                              cond=drifting_conditions)
    if use_cache and os.path.isfile(cache_file):
        d = np.load(cache_file, allow_pickle=False)
        if "sign_per_neuron" in d.files:
            z_train, z_test = d["z_train"], d["z_test"]
            if not time_last:                       # (B, N, T) -> (B, T, N)
                z_train, z_test = z_train.transpose(0, 2, 1), z_test.transpose(0, 2, 1)
            return (d["c_train"], z_train, d["c_test"], z_test,
                    d["area_per_neuron"], d["sign_per_neuron"])
        # Old cache (pre-signs) — fall through to recompute everything.

    AtoT = AllenToTensor(session_id=session_id, stimulus="drifting_gratings",
                         dt=dt, cache_root=cache_root, verbose=True)
    session = AtoT.get_session()
    unit_ids = np.asarray(AtoT.get_units_indices(area=areas))
    if len(unit_ids) == 0:
        raise RuntimeError(f"No QC-passing units in {areas} for session {session_id}")

    # Sort so neurons are grouped by area in the canonical order `areas` was
    # given (e.g. VISp first, VISrl next…) — shared with get_perturbation_trials.
    unit_ids, area_per_neuron = _area_sort_units(session, unit_ids, areas)

    # Fast-spiking waveform classification: +1 excitatory, -1 inhibitory,
    # 0 outlier. Re-order to match the area-sorted neuron order above.
    sign_unit_ids, signs, _waveforms = AtoT.get_unit_ids_signs_and_waveforms(
        area=areas)
    sign_lookup = dict(zip(np.asarray(sign_unit_ids).tolist(),
                            np.asarray(signs).tolist()))
    sign_per_neuron = np.array([sign_lookup[int(u)] for u in unit_ids],
                                dtype=np.int8)

    # --- Drifting gratings ---------------------------------------------------
    dg_table = AtoT.get_scene_presentations_of_stim()
    dg_starts = dg_table["start_time"].to_numpy()
    c_dg = _condition_codes(dg_table, drifting_conditions)
    z_dg = _bin_trials(session, unit_ids, dg_starts, n_bins, dt)

    # --- Spontaneous ---------------------------------------------------------
    spont = session.get_stimulus_table(stimulus_names="spontaneous")
    spont_starts = []
    for _, row in spont.iterrows():
        t0, t1 = float(row["start_time"]), float(row["stop_time"])
        n = int(np.floor((t1 - t0) / trial_duration_s))
        spont_starts.extend([t0 + k * trial_duration_s for k in range(n)])
    spont_starts = np.asarray(spont_starts)
    z_spont = _bin_trials(session, unit_ids, spont_starts, n_bins, dt)
    c_spont = np.zeros(len(spont_starts), dtype=int)

    # --- 70/30 stratified split over the in-distribution sets ---------------
    c_in = np.concatenate([c_dg, c_spont])
    z_in = np.concatenate([z_dg, z_spont], axis=0)
    rng = np.random.default_rng(seed)
    c_train, z_train, c_test, z_test = _stratified_split(c_in, z_in, train_frac, rng)

    if use_cache:
        np.savez(cache_file, c_train=c_train, z_train=z_train,
                 c_test=c_test, z_test=z_test,
                 area_per_neuron=area_per_neuron,
                 sign_per_neuron=sign_per_neuron)
    if not time_last:                               # (B, N, T) -> (B, T, N)
        z_train, z_test = z_train.transpose(0, 2, 1), z_test.transpose(0, 2, 1)
    return c_train, z_train, c_test, z_test, area_per_neuron, sign_per_neuron


# Allen FC optogenetic protocol — fast_pulses parameters (per the FC docs):
#   2.5 ms pulses at 10 Hz over 1 s (duration column).
_FAST_PULSE_ON_S = 0.0025
_FAST_PULSE_PERIOD_S = 0.1


def opto_waveform(stimulus_name, duration, level, n_bins, dt, pre_s=0.0):
    """Reconstruct i(t) for one opto trial as a length-`n_bins` array.

    The trial window starts `pre_s` seconds before LED onset; the LED waveform
    therefore lives in `[pre_s, pre_s + duration)` of the trial, and i(t)
    before that is exactly 0 (within-trial baseline).

    Allen FC protocol waveforms (see `optogenetic_stimulation_epochs`):
      - "pulse"        : single square pulse of duration ∈ {5, 10} ms at `level`
      - "fast_pulses"  : 2.5 ms square pulses at 10 Hz over 1 s
      - "raised_cosine": 0.5 * (1 - cos(2π t / duration)) over 1 s, scaled by `level`

    Pulses narrower than `dt` are widened to one bin so they remain visible.
    """
    sig = np.zeros(n_bins, dtype=np.float32)
    pre = int(round(pre_s / dt))
    if stimulus_name == "pulse":
        n_on = max(1, int(round(duration / dt)))
        sig[pre:pre + n_on] = level
    elif stimulus_name == "fast_pulses":
        n_on = max(1, int(round(_FAST_PULSE_ON_S / dt)))
        n_pulses = int(round(duration / _FAST_PULSE_PERIOD_S))
        for k in range(n_pulses):
            i0 = pre + int(round(k * _FAST_PULSE_PERIOD_S / dt))
            sig[i0:i0 + n_on] = level
    elif stimulus_name == "raised_cosine":
        n_on = min(n_bins - pre, int(round(duration / dt)))
        if n_on > 0:
            t_on = np.arange(n_on) * dt
            sig[pre:pre + n_on] = level * 0.5 * (1.0 - np.cos(2 * np.pi * t_on / duration))
    else:
        raise ValueError(f"Unknown opto stimulus_name: {stimulus_name!r}")
    return sig


_LEVELS = (1.3, 1.7, 2.0)
_WAVEFORMS = ("pulse", "fast_pulses", "raised_cosine")


def perturbation_condition_from_light(light):
    """Compute integer condition ids `(n_trials,)` from the per-trial light
    arrays alone (no metadata needed):

        0           : sham, i(t) ≡ 0
        1 + 3*w + l : opto, with w ∈ {0,1,2} for {pulse, fast_pulses,
                      raised_cosine} and l ∈ {0,1,2} for level {1.3, 1.7, 2.0}

    Waveform is identified by counting nonzero bins (≤ 2 → pulse,
    3..15 → fast_pulses, > 15 → raised_cosine). Level is read from the
    peak of i(t) and snapped to the nearest of (1.3, 1.7, 2.0).
    """
    light = np.asarray(light)
    n = light.shape[0]
    cid = np.zeros(n, dtype=int)
    levels = np.array(_LEVELS)
    for i in range(n):
        sig = light[i]
        if not np.any(sig):
            continue  # sham -> 0
        nz = int((sig > 0).sum())
        if nz <= 2:
            w = 0  # pulse
        elif nz <= 15:
            w = 1  # fast_pulses (~10 nonzero bins)
        else:
            w = 2  # raised_cosine
        l = int(np.argmin(np.abs(levels - float(sig.max()))))
        cid[i] = 1 + 3 * w + l
    return cid


def perturbation_condition_label(cid):
    """Human-readable label for an integer condition id."""
    if cid == 0:
        return "sham"
    cid -= 1
    w, l = divmod(cid, 3)
    return f"{_WAVEFORMS[w]} l={_LEVELS[l]}"


def _build_sham_segment_starts(opto, perturb_dur_s, sham_buffer_s):
    """Place control windows in the gaps between LED events inside the opto
    block, with a `sham_buffer_s` buffer on both sides of every LED-on period.
    Returns an array of segment start times.

    Construction:
      - exclusion_i = [t0_i - buf, t0_i + duration_i + buf]
      - merge overlapping exclusions
      - inside [opto_block_start, opto_block_end], the gaps between merged
        exclusions are LED-free and ≥ `sham_buffer_s` away from any LED edge
      - place segments back-to-back from each gap's left edge,
        floor(gap_length / perturb_dur_s) of them per gap.
    Resulting segments are strictly disjoint from every LED-on window and at
    least `sham_buffer_s` away from any LED edge on both sides.
    """
    starts = np.asarray(opto["start_time"], dtype=float)
    ends = starts + np.asarray(opto["duration"], dtype=float)
    ex = np.stack([starts - sham_buffer_s, ends + sham_buffer_s], axis=1)
    ex = ex[np.argsort(ex[:, 0])]
    merged = [ex[0].tolist()]
    for a, b in ex[1:]:
        if a <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    merged = np.asarray(merged)

    block_start = float(starts.min())
    block_end = float(ends.max())
    gaps, prev = [], block_start
    for a, b in merged:
        a = max(a, block_start); b = min(b, block_end)
        if a > prev:
            gaps.append((prev, a))
        prev = max(prev, b)
    if prev < block_end:
        gaps.append((prev, block_end))

    seg_starts = []
    for a, b in gaps:
        n = int(np.floor((b - a) / perturb_dur_s))
        seg_starts.extend(a + k * perturb_dur_s for k in range(n))
    return np.asarray(seg_starts)


def get_perturbation_trials(session_id=SESSION_ID, dt=DT,
                            areas=VISUAL_AREAS,
                            perturb_dur_s=PERTURB_DUR_S,
                            perturb_pre_s=PERTURB_PRE_S,
                            sham_buffer_s=SHAM_BUFFER_S,
                            cache_root=DEFAULT_CACHE_ROOT,
                            include_sham=True,
                            use_cache=True, time_last=True):
    """Returns ``(light, z, meta)`` for the perturbation evaluation set.

    `z` is `(n_trials, n_neurons, n_bins)` = (B, N, T) when `time_last=True`
    (default); pass `time_last=False` for (B, T, N). Neurons are area-sorted
    to match `load_data` so the axes align with `sign_per_neuron`.

    Each trial is `perturb_dur_s` seconds long. LED-on opto trials are
    aligned so the LED onset falls at `t = perturb_pre_s` into the trial:
    the trial spans `[t_onset - perturb_pre_s, t_onset + perturb_dur_s -
    perturb_pre_s)`.

    If `include_sham=True` (the default), in-block sham segments are appended
    after the opto trials. Sham segments are 1-s windows drawn from the gaps
    between LED events, with a `perturb_pre_s` buffer from any LED edge —
    they are by construction LED-free (light=0 throughout) and disjoint
    from every opto stimulation. They are matched to the opto trials in
    visual context (gray screen, opto block) and recording state.

    Outputs:
        light:  (n_trials, n_bins) float32 — LED intensity i(t) per trial.
        z:      (n_trials, n_neurons, n_bins) int8 — spike counts per bin.
        meta:   dict with per-trial labels:
            - 'kind':         'opto' or 'sham' (length n_trials)
            - 'stimulus_name': waveform name for opto trials, '' for sham
            - 'level':        LED level for opto trials, 0.0 for sham
            - 'duration':     LED-on duration for opto trials, 0.0 for sham
    """
    n_bins = int(round(perturb_dur_s / dt))
    cache_file = _cache_path(cache_root, "perturb",
                              sess=session_id, dt=dt, dur=perturb_dur_s,
                              pre=perturb_pre_s, buf=sham_buffer_s,
                              areas=areas, sham=int(include_sham), srt=1)
    if use_cache and os.path.isfile(cache_file):
        d = np.load(cache_file, allow_pickle=False)
        meta = {"kind": d["kind"], "stimulus_name": d["stimulus_name"],
                "level": d["level"], "duration": d["duration"],
                "condition": d["condition"]}
        z = d["z"] if time_last else d["z"].transpose(0, 2, 1)
        return d["light"], z, meta

    AtoT = AllenToTensor(session_id=session_id, stimulus="drifting_gratings",
                         dt=dt, cache_root=cache_root, verbose=True)
    session = AtoT.get_session()
    unit_ids = np.asarray(AtoT.get_units_indices(area=areas))
    if len(unit_ids) == 0:
        raise RuntimeError(f"No QC-passing units in {areas} for session {session_id}")
    # Same area-sort as load_data so the neuron axis aligns with sign_per_neuron.
    unit_ids, _ = _area_sort_units(session, unit_ids, areas)

    opto = session.optogenetic_stimulation_epochs
    n_opto = len(opto)

    # --- Opto trial windows (LED onset at t = perturb_pre_s) ----------------
    opto_starts = np.asarray(opto["start_time"]) - perturb_pre_s
    z_opto = _bin_trials(session, unit_ids, opto_starts, n_bins, dt)
    light_opto = np.zeros((n_opto, n_bins), dtype=np.float32)
    for i, (_, row) in enumerate(opto.iterrows()):
        light_opto[i] = opto_waveform(row["stimulus_name"],
                                      float(row["duration"]),
                                      float(row["level"]),
                                      n_bins, dt, pre_s=perturb_pre_s)
    kind = ["opto"] * n_opto
    stim = list(opto["stimulus_name"].astype(str))
    level = list(opto["level"].astype(float))
    duration = list(opto["duration"].astype(float))

    if not include_sham:
        meta = {"kind": np.array(kind), "stimulus_name": np.array(stim),
                "level": np.array(level), "duration": np.array(duration),
                "condition": perturbation_condition_from_light(light_opto)}
        z_opto = z_opto if time_last else z_opto.transpose(0, 2, 1)
        return light_opto, z_opto, meta

    # --- In-block sham segments (LED-free, `sham_buffer_s` from any LED) ---
    sham_starts = _build_sham_segment_starts(opto, perturb_dur_s, sham_buffer_s)
    if len(sham_starts) == 0:
        light = light_opto
        z = z_opto
    else:
        z_sham = _bin_trials(session, unit_ids, sham_starts, n_bins, dt)
        light_sham = np.zeros((len(sham_starts), n_bins), dtype=np.float32)
        light = np.concatenate([light_opto, light_sham], axis=0)
        z = np.concatenate([z_opto, z_sham], axis=0)
        kind.extend(["sham"] * len(sham_starts))
        stim.extend([""] * len(sham_starts))
        level.extend([0.0] * len(sham_starts))
        duration.extend([0.0] * len(sham_starts))

    meta = {"kind": np.array(kind), "stimulus_name": np.array(stim),
            "level": np.array(level), "duration": np.array(duration),
            "condition": perturbation_condition_from_light(light)}
    if use_cache:
        np.savez(cache_file, light=light, z=z, **meta)
    if not time_last:                               # (B, N, T) -> (B, T, N)
        z = z.transpose(0, 2, 1)
    return light, z, meta


def find_PV_neurons(session_id=SESSION_ID, dt=DT, cache_root=DEFAULT_CACHE_ROOT):
    """Identify PV (opto-driven inhibitory) neurons from the raised-cosine trials.

    "Ideal" ground truth derived from the perturbation itself: in the strongest
    raised_cosine trials, split time bins into the high-light phase (i(t) > 0.5 *
    max) vs the rest; a neuron is PV if its mean spiking in the high phase exceeds
    the rest by more than a small fraction of its variability. Returns a boolean
    mask of shape (n_neurons,), neuron-axis aligned with `load_data`.
    """
    light, z_pert, meta = get_perturbation_trials(session_id=session_id, dt=dt,
                                                  cache_root=cache_root,
                                                  time_last=False)  # (B, T, N)
    sel = meta["stimulus_name"] == "raised_cosine"
    sel &= meta["level"] == meta["level"][sel].max()                # strongest level
    print(f"Perturbation set: raised_cosine @ level {meta['level'][sel][0]:.1f} "
          f"({int(sel.sum())} trials)")
    z_pert = (z_pert[sel] > 0).astype(np.float32)

    light = light[sel]
    mask = light > 0.5 * light.max(0)

    PV_mask = z_pert[mask].mean(0) - z_pert[np.logical_not(mask)].mean(0) > 0.05 * np.std(z_pert)
    N = z_pert.shape[-1]
    assert PV_mask.shape == (N,), f"got dim {PV_mask.shape}"
    assert N * 0.15 < PV_mask.sum() < N * 0.4, f"Got {PV_mask.sum()} / {N} PV neurons "
    return PV_mask


def _first_drifting_idx(c):
    idx = np.where(c >= 1)[0]
    return int(idx[0]) if len(idx) else 0


def _plot_rasters_and_decoding(c_tr, z_tr, c_te, z_te, dt,
                               per_area_results, area_names, n_per_area,
                               out="figures/dataloader_summary.png"):
    """Figure 1: top row = example train/test rasters.
    Bottom row = per-area confusion matrices (with train/test accuracy).
    Perturbation trials are deliberately not shown here — see figure 2."""
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(16, 9), constrained_layout=True)
    sf_top, sf_bot = fig.subfigures(2, 1, height_ratios=[1, 1.2])

    # --- Top row: example rasters (train + test only) --------------------
    sf_top.suptitle(f"Example single-trial rasters — {sum(n_per_area)} neurons "
                    f"({', '.join(area_names)})")
    axes_t = sf_top.subplots(1, 2, sharey=True)

    i_tr = _first_drifting_idx(c_tr)
    i_te = _first_drifting_idx(c_te)
    cum = np.cumsum([0] + list(n_per_area))
    for ax, name, z, c_id in zip(
            axes_t,
            ["train", "test"],
            [z_tr[i_tr], z_te[i_te]],
            [c_tr[i_tr], c_te[i_te]]):
        ax.imshow(z, aspect="auto", cmap="binary",
                  extent=[0.0, z.shape[1] * dt, z.shape[0], 0])
        for b in cum[1:-1]:
            ax.axhline(b, color="r", lw=0.5, alpha=0.6)
        ax.set_title(f"{name} trial (c={c_id})")
        ax.set_xlabel("time (s)")
    axes_t[0].set_ylabel("neuron (grouped by area)")

    # --- Bottom row: per-area confusion matrices -------------------------
    valid = [r for r in per_area_results if r[4] is not None]
    sf_bot.suptitle("Condition decoder — per-area confusion matrix "
                    "(rows = true condition, cols = predicted)")
    axes_b = sf_bot.subplots(1, max(1, len(valid)), squeeze=False)[0]
    for ax, (area, n_units, train_acc, test_acc, cm, labels) in zip(axes_b, valid):
        im = ax.imshow(cm, vmin=0, vmax=1, cmap="viridis", aspect="auto")
        ax.set_title(f"{area}  n={n_units}\ntrain={train_acc:.2f}  "
                     f"test={test_acc:.2f}")
        ax.set_xticks(np.arange(len(labels)))
        ax.set_yticks(np.arange(len(labels)))
        ax.set_xticklabels(labels.astype(str), rotation=90, fontsize=6)
        ax.set_yticklabels(labels.astype(str), fontsize=6)
        ax.set_xlabel("predicted")
        ax.set_ylabel("true")
        sf_bot.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.savefig(out, dpi=120)
    print(f"Saved {out}")
    plt.show()


def _plot_perturbation(light, z, meta, dt, area_names, n_per_area,
                       perturb_pre_s=PERTURB_PRE_S,
                       out="figures/dataloader_perturbation.png"):
    """Figure 2: per-condition grid of trial-averaged activity + i(t).

    Restricted to sham + the strongest level (2.0) of each waveform — 4 columns.
    Top row: trial-averaged firing rate (Hz), neurons grouped by area.
    Bottom row: i(t) for one example trial of that condition."""
    import matplotlib.pyplot as plt

    cid = meta["condition"]
    # Keep only sham + the strongest level (l=2) of each waveform; the lower
    # levels are visually similar and clutter the grid.
    strongest = {0, 3, 6, 9}
    conditions = sorted(c for c in np.unique(cid).tolist() if c in strongest)
    n_cond = len(conditions)
    n_bins = z.shape[-1]
    n_neurons = z.shape[1]
    t0 = -perturb_pre_s
    t1 = n_bins * dt - perturb_pre_s
    extent = [t0, t1, n_neurons, 0]
    t_axis = np.arange(n_bins) * dt + t0

    # Trial-averaged PSTH per condition (shared color scale, capped at 100 Hz
    # so sustained / pulse-aligned peaks don't wash out the rest of the grid).
    avgs = [z[cid == c].mean(axis=0) / dt for c in conditions]
    counts = [int((cid == c).sum()) for c in conditions]
    vmax = 100.0
    # Light example: first trial of each condition.
    light_examples = [light[np.where(cid == c)[0][0]] for c in conditions]
    y_light_max = max(2.1, float(max(s.max() for s in light_examples) * 1.1))

    fig, axes = plt.subplots(2, n_cond, figsize=(2.6 * n_cond + 1.2, 7.5),
                             gridspec_kw={"height_ratios": [3, 0.7]},
                             sharex=True, squeeze=False)

    cum = np.cumsum([0] + list(n_per_area))
    for col, c in enumerate(conditions):
        ax_psth = axes[0, col]
        ax_i = axes[1, col]

        im = ax_psth.imshow(avgs[col], aspect="auto", cmap="viridis",
                            extent=extent, vmin=0, vmax=vmax)
        ax_psth.set_title(f"c={c} {perturbation_condition_label(c)}\n"
                          f"({counts[col]} trials)", fontsize=8)
        for b in cum[1:-1]:
            ax_psth.axhline(b, color="r", lw=0.4, alpha=0.5)
        ax_psth.axvline(0.0, color="cyan", lw=0.8, alpha=0.7)

        ax_i.plot(t_axis, light_examples[col], color="C0", lw=1.2)
        ax_i.fill_between(t_axis, 0, light_examples[col], color="C0", alpha=0.25)
        ax_i.set_ylim(-0.05, y_light_max)
        ax_i.margins(x=0)
        ax_i.axvline(0.0, color="cyan", lw=0.6, alpha=0.5)
        ax_i.set_xlabel("time from LED onset (s)", fontsize=7)
        ax_i.tick_params(axis="both", labelsize=6)

    # First column gets a y label and per-area annotations.
    axes[0, 0].set_ylabel("neuron (grouped by area)")
    axes[1, 0].set_ylabel("i(t)")
    for i, name in enumerate(area_names):
        mid = (cum[i] + cum[i + 1]) / 2
        axes[0, 0].text(-0.02, mid, name,
                        transform=axes[0, 0].get_yaxis_transform(),
                        va="center", ha="right", fontsize=7, color="r")

    # Single shared colorbar at the right.
    fig.subplots_adjust(right=0.93)
    cax = fig.add_axes([0.945, 0.32, 0.012, 0.55])
    fig.colorbar(im, cax=cax, label="trial-avg rate (Hz)")

    fig.suptitle("Perturbation set — per-condition trial-averaged activity "
                 "and example i(t) (t=0 = LED onset)", fontsize=10)
    plt.savefig(out, dpi=120, bbox_inches="tight")
    print(f"Saved {out}")
    plt.show()


if __name__ == "__main__":
    # Importing here so the file can be imported as a library without sklearn.
    from scripts.condition_decoder import fit_score_and_confusion

    print(f"Loading session {SESSION_ID}  dt={DT}s  trial={TRIAL_DURATION_S}s")
    print("First run will download the NWB from the Allen warehouse (a few GB).")

    # Per-area decoder loop. Stash the z arrays so we can concatenate them
    # along the neuron axis and reuse for the multi-area perturbation plot
    # without paying for Allen-SDK loading twice.
    per_area = []
    z_tr_blocks, z_te_blocks, z_pt_blocks = [], [], []
    n_per_area, area_names = [], []
    c_tr = c_te = None
    light = pmeta = None
    for area in VISUAL_AREAS[:3]:
        print(f"\n[{area}] loading and fitting decoder...")
        try:
            c_tr_a, z_tr_a, c_te_a, z_te_a, _ = load_data(areas=[area])
        except RuntimeError as e:
            print(f"  skipped: {e}")
            per_area.append((area, 0, None, None, None, None))
            continue
        n_units = z_tr_a.shape[1]
        train_acc, test_acc, cm, labels = fit_score_and_confusion(
            c_tr_a, z_tr_a, c_te_a, z_te_a)
        print(f"  n_units={n_units}  train={train_acc:.3f}  test={test_acc:.3f}  "
              f"chance={1.0/len(labels):.3f}")
        per_area.append((area, n_units, train_acc, test_acc, cm, labels))
        z_tr_blocks.append(z_tr_a); z_te_blocks.append(z_te_a)
        n_per_area.append(n_units); area_names.append(area)
        if c_tr is None:
            c_tr, c_te = c_tr_a, c_te_a
        # Perturbation set for this area (light + meta are area-independent;
        # only z_pt_a depends on the area's units).
        light_a, z_pt_a, meta_a = get_perturbation_trials(areas=[area])
        z_pt_blocks.append(z_pt_a)
        if light is None:
            light, pmeta = light_a, meta_a

    z_tr = np.concatenate(z_tr_blocks, axis=1)
    z_te = np.concatenate(z_te_blocks, axis=1)
    z_pt = np.concatenate(z_pt_blocks, axis=1)
    n_opto = int((pmeta["kind"] == "opto").sum())
    n_sham = int((pmeta["kind"] == "sham").sum())
    print(f"\ntrain:              c={c_tr.shape}  z={z_tr.shape}  "
          f"({TRIAL_DURATION_S:g}s trials)")
    print(f"test:               c={c_te.shape}  z={z_te.shape}  "
          f"({TRIAL_DURATION_S:g}s trials)")
    print(f"perturbation set:   z={z_pt.shape}  ({PERTURB_DUR_S:g}s trials, "
          f"LED onset at t={PERTURB_PRE_S:g}s)  "
          f"opto={n_opto}  sham={n_sham}")
    print(f"distinct conditions in train: {sorted(np.unique(c_tr).tolist())}")

    # Figure 1: rasters + per-area classifier accuracy. Blocks until closed.
    _plot_rasters_and_decoding(c_tr, z_tr, c_te, z_te, DT,
                               per_area, area_names, n_per_area)

    # Figure 2: in-block sham vs opto from the perturbation set (multi-area).
    _plot_perturbation(light, z_pt, pmeta, DT, area_names, n_per_area)
