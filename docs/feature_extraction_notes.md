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

**Full-cohort run.** Built in
`notebooks/07_feature_extraction_full_cohort.ipynb`. Subject list built
by globbing `derivatives_heog_off/` for `sub-*_restEC-epo.fif`
(`.rglob`), not from the cohort spreadsheet or folder presence alone -
guarantees the list matches what's actually loadable. 160/163 found,
matching the 3 known-missing-BDF subjects. Loop: log-and-continue, 0
failures. Assembled to (160, 146). Missingness: only `preprocessing_error`
null (expected - populated only on failure). Distribution: 143/160
(89.4%) show posterior > frontal alpha; the 17 non-matching subjects show
no meaningful clustering on `n_epochs_after` or `autoreject_extreme`
versus cohort baseline. Saved as parquet, not CSV, for exact dtype
round-tripping on repeated reload (`pyarrow` installed via pip; `conda
install` failed on a solver conflict with the pinned `python=3.11` env).
Reload-verified via `.equals()`.

**Ratio-power scope.** Three distinct things get called "ratio power":
relative power (band/total, a normalization), cross-band ratios like
theta/beta (ADHD literature only, no MDD/rTMS grounding - excluded), and
frontal asymmetry. Asymmetry excluded per Decision 5's existing citation
of van der Vinne et al. (2017)'s meta-analysis - a stronger source than
what was initially considered for inclusion. Relative power deferred, not
excluded: cheap to derive post-hoc from the saved absolute-power matrix,
no reason to compute before needed.

## PLI

Built and validated (pilot + 6-subject batch) in
`notebooks/06_feature_extraction_pilot.ipynb`, refactored into
`src/features.py`.

**Tool choice: `mne_connectivity.spectral_connectivity_epochs` (v0.8.1),
not a manual Hilbert-transform build.** Validated library implementation
judged lower-risk than re-deriving circular phase statistics from scratch
(phase-wrapping edge cases are easy to get subtly wrong).

**Pair structure: upper-triangle only.** 325 of 676 possible pairs (26
choose 2), via `itertools.combinations`. A full matrix would introduce
perfectly collinear duplicate columns - actively harmful for the planned
regularized linear models and nested feature selection, not just wasted
storage. `indices` must be passed explicitly - left at default (`None`),
the function returns the full n^2 matrix (confirmed from the docstring).

**Band scope: all 5 bands**, matching band power, per Decision 5's
compute-once-curate-later pattern. Alpha-band PLI has direct stability
evidence (Dominicus et al., 2025); other bands lack that grounding - to
flag explicitly when the primary arm's curated pool is finalized.

**Epoch aggregation.** Handled internally via cross-spectral density
accumulated across epochs (multitaper, 7 DPSS windows) - not a literal
per-epoch-then-mean loop as originally sketched during planning;
mathematically equivalent, stated precisely here to avoid overclaiming
the mechanism. Concatenating epochs into one continuous trace was
considered and rejected - epochs aren't contiguous (autoreject drops
some), so concatenation would introduce artificial phase discontinuities
at epoch boundaries.

**Refactor validation.** `compute_pli` reproduces all 1625
notebook-derived values exactly (0/1625 mismatches).

**Orchestrator.** Updated to call `compute_pli` alongside
`compute_band_power`. Batch-tested on 6 subjects, varying epoch counts:
6/6 succeeded, all values within [0,1], no NaNs.

**[0,1] range assertion added inside `compute_pli`**, not just checked ad
hoc - unlike band power, PLI has a hard mathematical property worth
enforcing every call. Same reasoning as preprocessing's beta-plausibility
guard: cheaper to fail loudly at computation than catch it downstream.

## Deferred

Full-cohort PLI, coherence, PLV. Kuramoto. `requirements.txt`/
`environment.yml` (currently empty; most dependencies are conda-installed
with local build-cache paths, not usable via plain `pip freeze`).