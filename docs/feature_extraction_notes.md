# Feature extraction notes

Implementation log for feature extraction: bugs caught and reasoning too
granular for `modelling_decisions.md` (pre-extraction decisions) or
`README.md` (high-level, batched per session).

## Band power

Built and validated in `notebooks/06_feature_extraction_pilot.ipynb`,
refactored into `src/features.py`. Validated on pilot subject
`sub-87999321` (restEC, heog_off).

**PSD method: Welch, not MNE's multitaper default.** `compute_psd()`
defaults to multitaper unless specified. Chang et al. (2025) states "FFT"
without specifying segmentation - ambiguous. Roelofs et al. (2021)'s
similarly-labeled "FFT" approach (4s segments, 50% overlap, Hamming
window) is Welch's method by construction; Stolz et al. (2023), a
separate TDBRAIN rTMS study, names Welch explicitly. Welch adopted on
this basis. Added to `modelling_decisions.md` Decision 2. Open item:
MNE's default window size (4.096s) not checked against literature.

**Units: V^2/Hz vs uV^2/Hz.** `compute_psd()` returns V^2/Hz (MNE stores
data in volts); literature reports uV^2/Hz. Caught by checking output
magnitude (delta power ~3.5e-11, implausible) rather than trusting the
code ran without error. Fixed with `* 1e12` at `band_mean`. Treated as a
correctness fix, not a modelling transform - no leakage risk, no
fold-dependence, so it belongs at extraction.

**Channel mismatch: `epochs.ch_names` vs `spectrum.ch_names`.**
`epochs.ch_names` returns 31 channels (includes EOG/Erbs/Mass/Status);
`spectrum.ch_names` returns the 26 actually used for PSD. Using the
former would have silently mislabeled every column. Caught by explicit
comparison before writing the pairing logic. `spectrum.ch_names` used
throughout.

**Band boundary double-counting.** Initial masks used `>=`/`<=` on both
ends, so a bin landing exactly on a shared boundary (e.g. 8.0 Hz) was
double-counted across adjacent bands. Fixed to half-open intervals
(`>= low & < high`).

**QC log column collision.** `batch_results_log_full_cohort.csv` has its
own `status`/`error` columns from preprocessing, colliding with the
orchestrator's `reason` field. Renamed on merge: `status` ->
`preprocessing_status`, `error` -> `preprocessing_error`.

**Refactor validation.** `compute_band_power` in `src/features.py`
reproduces notebook output exactly (`Fp1_delta_power`: matched to 15
decimal places).

**Orchestrator failure handling.** Tested against one success case and
one deliberate failure (nonexistent subject ID). `pd.DataFrame([...])`
correctly NaN-fills missing columns via pandas' default behavior - no
custom handling needed.

## Deferred

PLI, coherence, PLV, Kuramoto. Full-cohort run not yet started.