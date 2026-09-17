# EEG-Based Prediction of rTMS Treatment Response in Depression

## Summary

This project tests whether pre-treatment resting-state EEG features can predict rTMS response in major depressive disorder, using the TDBRAIN dataset (van Dijk et al., 2022; n=160-163). Framed as a predictive-enrichment problem, a biomarker-based patient-stratification concept used in CNS clinical trial design.

Seven pre-specified analyses were run: a primary feature (frontal alpha asymmetry, FAA), an extension (adding Kuramoto synchrony measures), two preprocessing sensitivity checks, a full feature-bank arm with nested selection and nonlinear classifiers, and two independent replication checks of published EEG biomarkers. All results are reported regardless of outcome, avoiding selective reporting.

**Headline finding:** FAA shows a weak but consistent association with responder status (elastic-net balanced accuracy 0.577-0.618 depending on preprocessing arm, p<0.05 in most configurations) that is robust to two preprocessing choices but not improved by adding synchrony features or expanding to the full ~5,000-feature bank with nonlinear models. Both independent biomarker replication checks (Bailey et al.'s theta connectivity; Roelofs et al.'s IAF-proximity) returned null results in this cohort.

**Stack:** Python, MNE-Python, scikit-learn, XGBoost. Full methodology and citations in [`docs/modelling_decisions.md`]; full build narrative in [`docs/development_log.md`]).

## Results at a glance

| Test | What's tested | Headline result | Significant? |
|---|---|---|---|
| Arm 1 | FAA alone (primary pool) | Elastic-net 0.587, p=0.016; other 3 classifiers at ~0.05 boundary | Partial (1/4) |
| Arm 2 | FAA + Kuramoto (order/metastability) | All 4 classifiers 0.597-0.618, p=0.008-0.025; no improvement over Arm 1 (paired test p=0.22-0.36) | Yes, but not better than Arm 1 |
| Arm 4 | Full feature bank (~5,015 features), nested MI selection, +XGBoost/RF | All 6 classifiers null, p=0.139-0.931; severe fold-to-fold instability | No |
| Arm 5 | FAA, heog_on (HEOG-correction sensitivity check) | All 4 classifiers 0.577-0.584, p=0.019-0.037 | Yes (4/4) |
| Arm 6 | FAA, restEO (resting-condition sensitivity check) | All 4 classifiers 0.608, p=0.005-0.007 | Yes (4/4) |
| Bailey supplementary | Theta connectivity replication (Bailey et al., 2019/2021) | All 4 classifiers null, p=0.92-0.98 | No |
| IAF-proximity supplementary | IAF-to-10Hz replication (Corlier 2019/Roelofs 2021), n=42 | Logistic p=0.721; Mann-Whitney p=0.546 | No |

*Arm 3 was retired during design (no independent-sample-grounded connectivity construct available for that slot) - see `docs/modelling_decisions.md`. Full detail on every arm, including the Arm 4 instability pattern, is in `docs/development_log.md`.*

## Limitations

- **Sample size**: n=163 overall, n=42 for the IAF-proximity subgroup. A single held-out test split would have very high variance at this n (Varoquaux, 2018), so nested cross-validation plus permutation testing is used in its place - there is no fully independent lockbox test set.
- **Baseline session assumption**: for the 8/163 subjects with two recorded sessions, `sessID == 1` is taken as the pre-treatment baseline. This is a documented assumption, not something the TDBRAIN data descriptor confirms explicitly (see `docs/modelling_decisions.md`).
- **Effect sizes are weak**: balanced accuracy throughout sits at 0.58-0.62, below some published single-cohort benchmarks (e.g. Provaznikova et al., 2025: AUC 0.75-0.81) - reported as found, not adjusted to look stronger.

## Repository guide

### Documents

**[`docs/modelling_decisions.md`]** - every design decision, with reasoning and citations, made *before* results were seen. Read this to understand *why* the project is shaped the way it is. Covers: primary/secondary feature pool selection (Decisions 1-4), the Bailey biomarker's exclusion from the primary pool and repositioning as a standalone replication check (Decision 5), Kuramoto synchrony rationale (Decision 6), classifier/metric/deconfounding choices (Decision 7), and the Arm 4 feature-selection methodology addendum.

**[`docs/development_log.md`]** - the full chronological build narrative, session by session: what was tried, what broke, what changed and why, in the order it happened. Read this for the *story* of the project - cohort validation, preprocessing pipeline development, feature extraction, and each modelling arm's build process and results, including bugs caught and how they were diagnosed.

**[`docs/feature_extraction_notes.md`]** - implementation-level detail for every feature extraction function (band power, PLI, coherence/PLV, Kuramoto, IAF-proximity, Bailey theta PLI): validation methodology, tool choices, bugs caught, full-cohort QC results. Read this if you want to know exactly how a specific feature was computed and verified.

**[`docs/preprocessing_notes.md`]** - the preprocessing pipeline in detail: documented deviations from the original TDBRAIN methodology, HEOG correction approach, known limitations.

### Notebooks (numbered in build order)

| Notebook | Contents |
|---|---|
| `00_mne_fundamentals_tutorial.ipynb` | MNE-Python tool-learning (ICA exploration) - not the final pipeline |
| `1_cohort_exploration.ipynb` | Cohort definition and validation: MDD filtering, baseline-session investigation, Responder-label verification |
| `02_preprocessing_pilot.ipynb` | Preprocessing methodology, built and validated on one pilot subject |
| `03_batch_test.ipynb` | Preprocessing validated on a 6-subject stratified batch |
| `04_full_cohort_run.ipynb` | Preprocessing run across the full 160-subject cohort |
| `05_modelling_prechecks.ipynb` | Pre-extraction assumption checks (protocol composition, epoch-count/age confound) |
| `06_feature_extraction_pilot.ipynb` | Feature extraction functions, built and validated on the pilot subject |
| `07_feature_extraction_full_cohort.ipynb` | Full-cohort feature extraction and matrix-level QC |
| `08_arm1_primary_pool.ipynb` | **Arm 1**: FAA alone |
| `09_arm2_primary_pool_kuramoto.ipynb` | **Arm 2**: FAA + Kuramoto |
| `10_arm5_arm6_feature_extraction.ipynb` | **Arm 5** (heog_on) and **Arm 6** (restEO) sensitivity checks |
| `11_bailey_supplementary.ipynb` | Bailey theta connectivity replication check |
| `12_iaf_proximity_supplementary.ipynb` | IAF-proximity replication check |
| `13_arm4_full_feature_bank.ipynb` | **Arm 4**: full feature bank, nested selection, XGBoost/RF |

### Source code

**`src/preprocessing.py`** - the validated preprocessing pipeline (twelve functions + `preprocess_subject` orchestrator).
**`src/features.py`** - validated feature extraction functions (`compute_band_power`, `compute_pli`, `compute_coherence_plv`, `compute_kuramoto`, `compute_iaf`) + `extract_subject_features` orchestrator.
**`src/modelling.py`** - `AgeDeconfounder`, `run_nested_cv`, `paired_comparison_test` - reused unchanged across Arms 1, 2, 5, 6, and the Bailey/IAF supplementary tests.

## Reproducing this

- **Data access**: raw TDBRAIN data is not included in this repository and this project is not clone-and-run. The dataset is provided by the Brainclinics Foundation under a Data Use Agreement that restricts redistribution - access must be requested independently (see Data, below).
- **Environment**: Python 3.11, conda environment `eeg-rtms`; see `requirements.txt` for dependencies (MNE-Python, MNE-Connectivity, scikit-learn, XGBoost, pandas, scipy).
- **Pipeline order**: `00` (tool-learning, not part of the final pipeline) -> `1` (cohort definition and validation) -> `02`-`04` (preprocessing) -> `05` (pre-extraction assumption checks) -> `06`-`07` (feature extraction) -> `08`-`13` (modelling arms and supplementary tests).

## Data

This project uses the TDBRAIN dataset (van Dijk et al., 2022), provided by the Brainclinics Foundation under a Data Use Agreement. Data is licensed under CC BY 4.0; accompanying preprocessing code is licensed under MIT.

Citation: van Dijk, H., van Wingen, G., Denys, D. et al. The two decades brainclinics research archive for insights in neurophysiology (TDBRAIN) database. Sci Data 9, 333 (2022). https://doi.org/10.1038/s41597-022-01409-z

Data source: https://brainclinics.com/resources/tdbrain-dataset

Raw data is not included in this repository (see `.gitignore`) and is subject to the terms of the TDBRAIN Data Use Agreement, including restrictions on redistribution and re-identification of subjects.