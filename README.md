# EEG-Based Prediction of rTMS Treatment Response in Depression

Investigating whether pre-treatment EEG connectivity and oscillator-synchrony features can predict rTMS response in MDD.

## Data

This project uses the TDBRAIN dataset (van Dijk et al., 2022), provided by the Brainclinics Foundation under a Data Use Agreement. Data is licensed under CC BY 4.0; accompanying preprocessing code is licensed under MIT.

Citation: van Dijk, H., van Wingen, G., Denys, D. et al. The two decades brainclinics research archive for insights in neurophysiology (TDBRAIN) database. Sci Data 9, 333 (2022). https://doi.org/10.1038/s41597-022-01409-z

Data source: https://brainclinics.com/resources/tdbrain-dataset

Raw data is not included in this repository (see .gitignore) and is subject to the terms of the TDBRAIN Data Use Agreement, including restrictions on redistribution and re-identification of subjects.

## Cohort Definition and Validation Update (09/07/26)

### Subject selection

Filtered to `indication == 'MDD'` (pure diagnosis, excluding comorbid rows e.g. `MDD/OCD/ADHD`) with non-null `Responder` label. Final cohort: **163 unique subjects** (94 responders, 69 non-responders). Subjects with MDD diagnosis but no documented rTMS outcome (n=~159) and non-MDD diagnoses with rTMS outcomes were excluded as out of scope for this supervised prediction task.

### Baseline EEG session assumption

Of the 163 subjects, 155 have a single session; 8 have two sessions. Where multiple sessions exist, `sessID == 1` is taken as the pre-treatment baseline recording. **This is an assumption, not an explicitly confirmed rule**, the TDBRAIN data descriptor (van Dijk et al., 2022) does not state that session numbering corresponds to treatment chronology. Investigation ruled out an alternative explanation (that `Responder` was null for `ses-2` rows); `Responder` is in fact identical across a subject's sessions, so `sessID == 1` was a deliberate choice.

Confirmed all 163 subjects have both restEC and restEO baseline recordings available in the Discovery dataset.

### Responder definition - empirically verified

The `Responder` label was independently verified against raw BDI-II scores rather than trusted at face value. Percent improvement was computed as `-(BDI_post - BDI_pre) / BDI_pre * 100`, and a ≥50% threshold was compared against the existing label: **163/163 exact match.** This confirms `Responder` is defined as ≥50% reduction in BDI-II from baseline to post-treatment, consistent with the DLPFC-rTMS sample criteria described in van Dijk et al. (2022).

### Demographic and clinical balance checks (Responder vs Non-responder)

- **Age**: significant difference (M=42.85 vs 48.68, Welch's t=-2.68, p=.01). Responders are younger on average. **Flagged as a potential confound** for EEG-based modelling, since EEG features are known to vary with age (see TDBRAIN iAPF maturation findings). To be addressed in modelling (e.g. as a covariate or via stratification).
- **Gender**: balanced, no significant difference (χ²=1.44, p=.23).
- **Baseline BDI severity**: balanced, no significant difference (t=-1.45, p=.15), rules out baseline severity as a confound for treatment outcome.

### Missingness

No missingness in the fields used by this project (age, gender, BDI_pre, BDI_post, Responder) for the final 163-subject cohort - confirmed as a byproduct of the checks above. Broader spreadsheet fields (education, NEO-FFI, etc.) are out of scope for this project and were not audited, since they are not planned as model inputs.

## Update regarding preprocessing pipeline

The original plan was to use the TDBRAIN authors' published preprocessing pipeline directly. On inspection, this code does not run on the current dataset release: it expects CSV files with a fixed 33-channel layout and a legacy filename convention, neither of which match the BDF/BIDS-formatted files provided in the current TDBRAIN V3.1 dataset. This is a compatibility gap between the published code and the dataset's more recent format update, not a limitation of the methodology itself.

To address this, preprocessing for this project reimplements the documented methodology from van Dijk et al. (2022) natively in MNE-Python, rather than using the original code directly. Specifically, EOG artifact correction uses the regression-based method published by Gratton et al. (1983), matching the authors' documented approach, the ICA-based artifact removal explored in the MNE fundamentals notebook (`00_mne_fundamentals_tutorial.ipynb`) was tool-learning, not the method used in the final pipeline.

## Preprocessing status (17/08/26) - complete

Pipeline order (matching authors' dataset class methods): bipolarEOG -> demean -> apply_filters -> correct_EOG -> epoching -> artefact rejection.

Methodology developed and validated on a single pilot subject in `notebooks/02_preprocessing_pilot.ipynb`, refactored into `src/preprocessing.py` (twelve functions plus a `preprocess_subject` orchestrator), and validated at increasing scale: pilot subject, a 6-subject stratified batch (`notebooks/03_batch_test.ipynb`), and the full 160-subject usable cohort (`notebooks/04_full_cohort_run.ipynb`).

Full-cohort results: 160 of 163 subjects usable (3 have no source data present, not a pipeline issue). Two parallel output variants produced (`data/derivatives_heog_off/`, `data/derivatives_heog_on/`) for a planned modelling-stage sensitivity check on HEOG correction. autoreject parameter instability, investigated and characterised earlier, confirmed at full scale (~38% of subject/conditions flagged) and captured as QC metadata rather than excluded. Epoch retention: median 95.8%.

Known limitations, documented rather than silently resolved: HEOG correction confidence is improved (baseline-drift removal, literature-grounded duration bounds) but not fully resolved - HEOG correction is off by default. Full detail, all documented deviations from the authors' code, and the complete decision trail: see `docs/preprocessing_notes.md`.

## Modelling Plan Update (22/08/26)

Seven pre-extraction modelling decisions were finalised. Full detail, sources, and open items are documented in `docs/modelling_decisions.md`; summary below.

Two prechecks were run first (`notebooks/05_modelling_prechecks.ipynb`) to verify assumptions before they could be silently carried into feature extraction: cohort rTMS protocol composition against the literature basis for one candidate feature, and retained-epoch-count against age, to rule out a hidden selection effect from the epoch-count inclusion floor.

**Preprocessing/feature-scope comparison arms:** `heog_off` and `restEC` are primary for all main analyses; `heog_on` and `restEO` are dedicated sensitivity arms, since the full cohort was preprocessed under both conditions specifically to support this comparison.

**Features:** Frequency bands matched to the closest directly comparable study (Chang et al., 2025). PLI is the primary connectivity metric, chosen for volume-conduction robustness (Stam, Nolte & Daffertshofer, 2007), with coherence and PLV as secondary comparisons. Band/ratio power features are log-transformed; connectivity features are not. Subjects are aggregated by mean across epochs; the ~10-epoch inclusion floor was empirically confirmed non-age-biased and non-binding for this cohort (all 160 subjects with valid QC data retain 21-24 epochs).

**Modelling - three pre-specified arms, all reported regardless of outcome:**
- *Primary*: Ledoit-Wolf shrinkage LDA, elastic-net logistic regression, and Bayesian logistic regression on a literature-curated, replication-weighted feature pool (n=163).
- *Secondary/exploratory*: adds XGBoost and random forest on the full feature bank (~1,755+ features), with nested feature selection inside cross-validation.
- *Supplementary*: a replication check (not a trained classifier) of individual alpha frequency (IAF) proximity to 10Hz, restricted to the n=42 subjects on the matching rTMS protocol, since the published evidence for this feature (Corlier et al., 2019; Roelofs et al., 2021) only covers that specific subgroup.

Age is regressed out of features within training folds only, at the modelling stage, consistent across all arms.

**Open item:** an unresolved discrepancy was found between this cohort's rTMS protocol composition and the published TDBRAIN data descriptor (van Dijk et al., 2022, Table 2) - see `docs/modelling_decisions.md`, Decision 5, for detail. Flagged as a candidate for direct follow-up with Brainclinics if it becomes material to results.

## Feature extraction status (23/08/26)

Pipeline: `load_subject_epochs -> get_subject_qc -> compute_band_power ->
compute_pli`, tied together by `extract_subject_features` (orchestrator,
mirrors `preprocess_subject`'s structure).

**Band power** built and validated on pilot subject `sub-87999321`
(restEC, heog_off) in `notebooks/06_feature_extraction_pilot.ipynb`,
refactored into `src/features.py`. 130 columns (26 channels x 5 bands,
Chang et al. 2025 boundaries), PSD via Welch's method, values in
uV^2/Hz. Refactor validated against notebook output to 15 decimal places.

**Band power, full cohort** run in
`notebooks/07_feature_extraction_full_cohort.ipynb`: 160/160 subjects
succeeded (0 failures), assembled to a 160x146 feature matrix, QC'd
(missingness explained; 143/160 subjects show posterior > frontal alpha,
consistent with the pilot subject and expected topography), and saved as
`data/features/bandpower_full_cohort.parquet`. Ratio-power scope decided:
frontal asymmetry excluded per the existing Decision 5 citation (van der
Vinne et al., 2017); theta/beta excluded (ADHD literature only, no
MDD/rTMS grounding); relative power deferred, derivable post-hoc from the
saved matrix at low cost.

**PLI** (primary connectivity metric) built and validated on the pilot
subject and a 6-subject batch, using
`mne_connectivity.spectral_connectivity_epochs`. 1,625 columns (325
upper-triangle channel pairs x 5 bands). Refactor validated exactly
against notebook output (0/1625 mismatches); batch check (6 subjects,
varying epoch counts) passed with 0 failures, all values in [0,1], no
NaNs.

Orchestrator tested throughout against one success case and one
deliberate failure case, confirming clean NaN-filling when rows are
concatenated.

For full implementation detail (bugs caught, unit conversion, channel
mismatch, tool choices, and reasoning), see
[`docs/feature_extraction_notes.md`](docs/feature_extraction_notes.md).

Full-cohort PLI, coherence, PLV, and Kuramoto are next.