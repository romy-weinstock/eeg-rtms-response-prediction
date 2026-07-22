# EEG-Based Prediction of rTMS Treatment Response in Depression

   Investigating whether pre-treatment EEG connectivity and oscillator-synchrony features can predict rTMS response in MDD, framed as a predictive-enrichment problem for CNS drug development.

## Data

This project uses the TDBRAIN dataset (van Dijk et al., 2022), provided by the Brainclinics Foundation under a Data Use Agreement. Data is licensed under CC BY 4.0; accompanying preprocessing code is licensed under MIT.

Citation: van Dijk, H., van Wingen, G., Denys, D. et al. The two decades brainclinics research archive for insights in neurophysiology (TDBRAIN) database. Sci Data 9, 333 (2022). https://doi.org/10.1038/s41597-022-01409-z

Data source: https://brainclinics.com/resources/tdbrain-dataset

Raw data is not included in this repository (see .gitignore) and is subject to the terms of the TDBRAIN Data Use Agreement, including restrictions on redistribution and re-identification of subjects.

## Update regarding preprocessing pipeline

The original plan was to use the TDBRAIN authors' published preprocessing pipeline directly. On inspection, this code does not run on the current dataset release: it expects CSV files with a fixed 33-channel layout and a legacy filename convention, neither of which match the BDF/BIDS-formatted files provided in the current TDBRAIN V3.1 dataset. This is a compatibility gap between the published code and the dataset's more recent format update, not a limitation of the methodology itself.

To address this, preprocessing for this project reimplements the documented methodology from van Dijk et al. (2022) natively in MNE-Python, rather than using the original code directly. Specifically, EOG artifact correction uses the regression-based method published by Gratton et al. (1983), matching the authors' documented approach, the ICA-based artifact removal explored in the MNE fundamentals notebook (`00_mne_fundamentals_tutorial.ipynb`) was tool-learning, not the method used in the final pipeline.

## Cohort Definition and Validation Update (09/07/26)
### Subject selection
Filtered to `indication == 'MDD'` (pure diagnosis, excluding comorbid rows e.g. `MDD/OCD/ADHD`)
with non-null `Responder` label. Final cohort: **163 unique subjects** (94 responders, 69 non-responders).
Subjects with MDD diagnosis but no documented rTMS outcome (n=~159) and non-MDD diagnoses with rTMS outcomes were excluded as out of scope for this supervised prediction task.

### Baseline EEG session assumption
- Of the 163 subjects, 155 have a single session; 8 have two sessions. Where multiple sessions exist,`sessID == 1` is taken as the pre-treatment baseline recording. **This is an assumption, not an explicitly confirmed rule**, the TDBRAIN data descriptor (van Dijk et al., 2022) does not state that session numbering corresponds to treatment chronology. Investigation ruled out an alternative explanation (that `Responder` was null for `ses-2` rows);`Responder` is in fact identical across a subject's sessions, so `sessID == 1` was a deliberate choice. 
- Confirmed all 163 subjects have both restEC and restEO baseline recordings
available in the Discovery dataset.

### Responder definition - empirically verified
The `Responder` label was independently verified against raw BDI-II scores rather than trusted at face value. Percent improvement was computed as `-(BDI_post - BDI_pre) /BDI_pre * 100`, and a ≥50% threshold was compared against the existing label: **163/163 exact match.** This confirms `Responder` is defined as ≥50% reduction in BDI-II from baseline to post-treatment, consistent with the DLPFC-rTMS sample criteria described in van Dijk et al. (2022).

### Demographic and clinical balance checks (Responder vs Non-responder)
- **Age**: significant difference (M=42.85 vs 48.68, Welch's t=-2.68, p=.01). Responders are younger on average. **Flagged as a potential confound** for EEG-based modelling, since EEG features are known to vary with age (see TDBRAIN iAPF maturation findings). To be addressed in modelling (e.g. as a covariate or via stratification).
- **Gender**: balanced, no significant difference (χ²=1.44, p=.23).
- **Baseline BDI severity**: balanced, no significant difference (t=-1.45, p=.15), rules out baseline severity as a confound for treatment outcome.

### Missingness
No missingness in the fields used by this project (age, gender, BDI_pre, BDI_post, Responder) for the final 163-subject cohort - confirmed as a byproduct of the checks above. Broader spreadsheet fields (education, NEO-FFI, etc.) are out of scope for this project and were not audited, since they are not planned as model inputs.

## Preprocessing status

Pipeline order (matching authors' `dataset` class methods): `bipolarEOG -> demean -> apply_filters -> correct_EOG`.

Completed and validated on pilot subject:
- `bipolarEOG`
- `demean`
- `apply_filters`
- `correct_EOG` (VEOG): artefact detection, segment padding, amplitude- and duration-plausibility guards validated. Gratton regression generalised across all 26 EEG channels and all 6 guarded restEO segments, validated against a single-channel proof-of-concept and expected scalp physiology.
- `correct_EOG` (HEOG): artefact detection (with a z-scoring fix required for reliable detection), amplitude- and duration-plausibility guards, and Gratton regression generalised across all 26 EEG channels and all 7 guarded restEO segments, regressed onto the VEOG-corrected data per the authors' documented sequential order. A beta-plausibility guard excludes implausible per-segment coefficients arising from a known short-window regression limitation.

For full methodology detail, documented deviations from the authors' code, and open items, see [`docs/preprocessing_notes.md`](docs/preprocessing_notes.md).