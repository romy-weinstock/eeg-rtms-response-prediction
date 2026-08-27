# Feature extraction notes

Implementation log for feature extraction: bugs caught and reasoning too granular for `modelling_decisions.md` (pre-extraction decisions) or `README.md` (high-level, batched per session).

## Band power

Built and validated in `notebooks/06_feature_extraction_pilot.ipynb`, refactored into `src/features.py`. Validated on pilot subject `sub-87999321` (restEC, heog_off).

**PSD method: Welch, not MNE's multitaper default.** `compute_psd()` defaults to multitaper unless specified. Chang et al. (2025) states "FFT" without specifying segmentation - ambiguous. Roelofs et al. (2021)'s similarly-labeled "FFT" approach (4s segments, 50% overlap, Hamming window) is Welch's method by construction; Stolz et al. (2023), a separate TDBRAIN rTMS study, names Welch explicitly. Welch adopted on this basis. Added to `modelling_decisions.md` Decision 2. Open item: MNE's default window size (4.096s) not checked against literature.

**Units: V^2/Hz vs uV^2/Hz.** `compute_psd()` returns V^2/Hz (MNE stores data in volts); literature reports uV^2/Hz. Caught by checking output magnitude (delta power ~3.5e-11, implausible) rather than trusting the code ran without error. Fixed with `* 1e12` at `band_mean`. Treated as a correctness fix, not a modelling transform - no leakage risk, no fold-dependence, so it belongs at extraction.

**Band boundary double-counting.** Initial masks used `>=`/`<=` on both ends, so a bin landing exactly on a shared boundary (e.g. 8.0 Hz) was double-counted across adjacent bands. Fixed to half-open intervals (`>= low & < high`).

**QC log column collision.** `batch_results_log_full_cohort.csv` has its own `status`/`error` columns from preprocessing, colliding with the orchestrator's `reason` field. Renamed on merge: `status` -> `preprocessing_status`, `error` -> `preprocessing_error`.

**Refactor validation.** `compute_band_power` in `src/features.py` reproduces notebook output exactly (`Fp1_delta_power`: matched to 15 decimal places).

**Orchestrator failure handling.** Tested against one success case and one deliberate failure (nonexistent subject ID). `pd.DataFrame([...])` correctly NaN-fills missing columns via pandas' default behavior - no custom handling needed.

**Full-cohort run.** Built in `notebooks/07_feature_extraction_full_cohort.ipynb`. Subject list built
by globbing `derivatives_heog_off/` for `sub-*_restEC-epo.fif` (`.rglob`), not from the cohort spreadsheet or folder presence alone - guarantees the list matches what's actually loadable. 160/163 found, matching the 3 known-missing-BDF subjects. Loop: log-and-continue, 0 failures. Assembled to (160, 146). Missingness: only `preprocessing_error` null (expected - populated only on failure). Distribution: 143/160 (89.4%) show posterior > frontal alpha; the 17 non-matching subjects show no meaningful clustering on `n_epochs_after` or `autoreject_extreme` versus cohort baseline. Saved as parquet, not CSV, for exact dtype round-tripping on repeated reload (`pyarrow` installed via pip; `conda install` failed on a solver conflict with the pinned `python=3.11` env). Reload-verified via `.equals()`.

## PLI

Built and validated (pilot + 6-subject batch) in `notebooks/06_feature_extraction_pilot.ipynb`, refactored into `src/features.py`.

**Pair structure: upper-triangle only** (325/676), via `itertools.combinations` - avoids collinear duplicate columns. `indices`
must be passed explicitly, or the function returns the full n^2 matrix.

**Band scope: all 5 bands**, per Decision 5's compute-once-curate-later pattern. Alpha has direct stability evidence (Dominicus et al., 2025); other bands don't.

**Epoch aggregation.** Internal, via cross-spectral density accumulated across epochs - not a literal per-epoch-then-mean loop as originally planned; mathematically equivalent, stated precisely to avoid
overclaiming.

**Refactor validation.** `compute_pli` reproduces all 1625 notebook-derived values exactly (0/1625 mismatches).

**Orchestrator.** Updated to call `compute_pli` alongside `compute_band_power`. Batch-tested on 6 subjects, varying epoch counts: 6/6 succeeded, all values within [0,1], no NaNs. **[0,1] range assertion added inside `compute_pli` itself**, not just checked ad hoc - hard mathematical property worth enforcing every call, same reasoning as preprocessing's beta-plausibility guard.

**Full-cohort run.** Extended `07` (no longer band-power-only). 160/160, 0 failures. Assembled to (160, 1771). Missingness/range clean, same pattern as band power. 

## Coherence and PLV

Built and validated (pilot + 6-subject batch) in `notebooks/06_feature_extraction_pilot.ipynb`, refactored into `src/features.py`. 

**Single-call tooling: `spectral_connectivity_epochs(method=['coh','plv'], ...)`.** Computes both metrics together, returning a list of `SpectralConnectivity` objects in the order passed - confirmed via `.method` on each object, not assumed from the docstring. PLI kept separate to avoid recomputing an already-validated metric.

**Same scope as PLI**: all 5 bands, upper-triangle pairs only.

**Illustrative check (not validation evidence).** Fp1-Fp2, alpha, pilot subject: coherence 0.947, PLV 0.993, vs. PLI 0.137 for the same pair/band. Demonstrates Decision 2's rationale directly - coherence/PLV lack PLI's volume-conduction correction and inflate for adjacent electrodes; PLI correctly suppresses it. Single-subject, single-pair - illustrative only.

**Refactor validation.** `compute_coherence_plv` validated against an independent manual derivation  (`spectral_connectivity_epochs(method=['coh', 'plv'], ...)` called directly in the notebook). Full equality check across all 3250 columns: 0 mismatches, matching PLI's 0/1625 standard.

**Orchestrator.** Updated to call `compute_coherence_plv` alongside `compute_band_power` and `compute_pli`. Batch-tested on 6 subjects: 6/6 succeeded, all 3250 coh/plv values within [0,1], no NaNs.

**Full-cohort run.** Extended `07` notebook. 160/160, 0 failures. Assembled to (160, 5021). Missingness/range clean, same pattern as for band power and pli. 

## Kuramoto order parameter and metastability

Built and validated (pilot + 6-subject batch + full cohort) in `notebooks/06_feature_extraction_pilot.ipynb`, refactored into
`src/features.py`.

**Method**: band-pass filter each of the five bands, Hilbert transform to extract instantaneous phase, order parameter = mean of R(t) over the epoch, metastability = std of R(t) over the epoch. Global (all 26 channels), not sub-grouped. Full rationale in `modelling_decisions.md`, Decision 6.

**Filter edge-effect check.** Full-epoch vs. trimmed-epoch order parameter/metastability compared across all 23 retained epochs, pilot subject, delta (worst case) and gamma (best case). Diffs small relative to epoch-to-epoch spread and alternating in sign rather than systematic - no trimming applied.

**Refactor validation.** Manual derivation (independent of `compute_kuramoto`) vs. refactored output: 0/10 mismatches.

**Orchestrator.** Updated to call `compute_kuramoto` alongside the other three functions. 

**Full-cohort run.** Extended `07` notebook. 160/160 succeeded (confirmed via loop failure count, `preprocessing_status`, and missingness pattern), 160x5031 final shape, all Kuramoto values within expected bounds, no NaNs. Saved to `full_cohort_features.parquet`, reload-verified.

## IAF-proximity (supplementary, protocol-1 only)

Built and validated (manual derivation + refactor) in `notebooks/06_feature_extraction_pilot.ipynb`, refactored into
`src/features.py`, extracted for all 42 protocol-1 subjects in `notebooks/07_feature_extraction_full_cohort.ipynb`. Not part of the primary or secondary feature bank; a standalone supplementary analysis per Decision 5.

**Method**: Welch PSD (consistent with this project's reading of Roelofs et al. (2021)'s "FFT" method, Decision 2), peak frequency within 7-13 Hz (not this project's primary 8-13 Hz alpha band - a deliberate exception for replication fidelity) at a single electrode, F3, matched to the 10 Hz left-DLPFC stimulation site. IAF-prox = |IAF - 10|.

**Documented deviations from Roelofs et al. (2021)**: PSD computed from already-epoched, already-QC'd ~5s epochs rather than their continuous 4s/50%-overlap segmentation.

**Peak identifiability check.** Roelofs et al. excluded ~12.6% of their sample (n~175) for lacking a dominant alpha peak, using a 1.5 Z-score cutoff on peak power, justified by a visible bimodal distribution in their sample. Checked whether this project's n=42 sample shows the same pattern: peak power distribution is unimodal and right-skewed, with no distinguishable low-power subgroup; 0/42 subjects fall below their -1.5 Z threshold. The empirical basis for their exclusion criterion is not
present in this sample, so no subjects were excluded; all 42 retained.

**Extraction.** Standalone loop over the 42 protocol-1 subject IDs, not integrated into `extract_subject_features` (which assumes uniform extraction across all 160 subjects). 42/42 succeeded, no NaNs, all IAF
values within [7,13]. Saved to `data/features/iaf_protocol1.parquet`, reload-verified.

**Association test** (Decision 5, Decision 5 addendum): deferred to the modelling stage (Day 10), not run here. Planned: logistic regression (IAF-prox + age) as primary, Mann-Whitney U (IAF-prox only) as a companion check, given the protocol-1 subgroup's binary responder outcome (17 non-responders / 25 responders) doesn't map onto Roelofs et al.'s continuous-outcome correlation design.

## Full feature matrix assembly and QC

**Assembly.** No merge required. `extract_subject_features` was extended incrementally through this notebook (band power -> +PLI -> +coherence/PLV -> +Kuramoto), and each extension re-ran extraction across the full cohort rather than appending to prior output. `full_cohort_features.parquet` therefore already contains all four families; `bandpower_full_cohort.parquet` is a superseded pre-PLI snapshot, not a separate piece needing integration. 

**`feature_cols` vs. identity/QC columns.** Defined an explicit 16-column identity/QC list (`subject_id`, `preprocessing_status`, `autoreject_consensus`, etc.) once, so downstream variance/CV checks run against the 5015 genuine feature columns rather than being contaminated by constant bookkeeping columns (`preprocessing_status`, `preprocessing_error`) that would trivially flag as zero-variance for uninteresting reasons.

**Matrix-level QC, run in addition to (not instead of) each family's own per-family QC:**
- Row-wise missingness: 1 NaN/subject uniformly, all `preprocessing_error` (expected, null-by-design on success).
- Exact-zero variance: 0/5015 columns.
- CV (std/mean) near-zero check: no natural gap in the distribution (checked via zoomed histogram, CV < 0.1) — lowest 30 CV columns are exclusively PLV, exclusively spatially-adjacent electrode pairs. Extends Decision 2's single-subject illustrative volume-conduction comparison (Fp1-Fp2, coherence 0.947/PLV 0.993 vs. PLI 0.137) to a full-cohort, multi-pair confirmation. No exclusions applied — effect is graded, not binary; any down-weighting or exclusion is a modelling-stage call, not extraction-stage.