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
- `correct_EOG` (VEOG): artefact detection, segment padding, amplitude-plausibility guard, and
  a duration-plausibility guard (see deviations below) validated. Gratton regression generalised
  across all 26 EEG channels and all 6 amplitude- and duration-guarded restEO segments
  (per-(channel, segment) coefficient estimation via unweighted least-squares, matching authors;
  tukey-tapered correction weight; subtraction scoped to each flagged segment only). Validated
  against the single-channel proof-of-concept (exact coefficient match, Fp1) and against expected
  scalp physiology (mean coefficient by channel shows a monotonic frontal-to-posterior gradient,
  strongest at Fp1/Fp2, near zero at occipital sites).
- `correct_EOG` (HEOG): not yet implemented. Detection pipeline will follow the same structure as
  VEOG (lowpass -> Hilbert envelope -> boxcar smoothing -> threshold -> segment collapse -> padding
  -> guards), but amplitude and duration thresholds require independent justification rather than
  reuse of VEOG's values, since saccade amplitude scales with gaze angle (unlike blink amplitude,
  which is comparatively consistent) and saccades are typically shorter in duration than blinks.

### restEC artefact segments found to be non-ocular

Both VEOG segments originally detected in the pilot subject's restEC (eyes-closed) recording 
were found, on inspection, to be step/pop artefacts rather than genuine blinks: peak-to-peak 
amplitude of 11,404.8 µV and 10,234.7 µV respectively, roughly one to two orders of magnitude 
above a physiologically plausible blink (typically tens to a few hundred µV). Confirmed present in 
the pre-`apply_filters` data, ruling out filtering as the cause. One of these segments coincides 
with the F3 discontinuity noted below near the recording's end; the other coincides with two of 
the three F3 discontinuities near samples ~14,000-16,500. Both restEC segments were excluded 
by the amplitude-plausibility guard.

The subject's restEO (eyes-open) recording was used instead for the regression proof-of-concept, 
since eyes-open resting state reliably contains genuine blinks: 8 candidate segments were 
detected, of which 7 passed the amplitude guard (peak-to-peak 67.9-576.6 µV) and 1 was 
excluded (18,162.1 µV, again near the recording's end).

**Known open items**:
- Three step-like discontinuities in channel F3 (two near samples ~14,000-16,500, one near the 
  recording's end, ~sample 60,000), flagged during `apply_filters` validation. Confirmed to 
  coincide with the restEC step-artefact segments described above; not yet addressed with a 
  dedicated jump/step-artefact detector.
- Segments near the end of the recording appear artefact-prone in both restEC and restEO for 
  this subject (possibly equipment settling or cap removal); not yet investigated, noted as a 
  pattern to watch across subjects.
- HEOG amplitude and duration guard thresholds need their own justification (literature-based or
  empirical) rather than reuse of VEOG's values, given the differing physiology of saccades vs.
  blinks.
- Open design decision: how subjects with zero valid (post-guard) segments in a given condition 
  (as occurred for this subject's restEC recording) should be represented downstream - e.g. 
  channel data left unchanged where no correction is possible.

Documented deviations from authors' code (library compatibility, no behaviour change):
- `np.int` -> `int()` (deprecated in current NumPy)
- `scipy.signal.boxcar` -> `scipy.signal.windows.boxcar` (relocated in current SciPy)

Documented deviations from authors' code (behaviour change, justified):
- One-sided z-score threshold (`z > threshold`) used in VEOG artefact detection, in place of the 
  authors' two-sided condition, since the input (`boxdata`, an amplitude envelope) is non-negative 
  by construction; the two-sided condition false-flagged ~76% of the recording as artefact due to 
  distribution skew.
- Amplitude-plausibility guard added before the Gratton regression, excluding segments with 
  implausible peak-to-peak VEOG amplitude. The authors' own pipeline (`autopreprocess_pipeline.py`) 
  runs `correct_EOG` before `detect_jumps`, with no equivalent safeguard - meaning step/pop 
  artefacts would be regressed against as if ocular in the original published method. This guard is 
  a stopgap addressing that specific gap, not a claim of perfectly separating artefact types, 
  pending a proper jump-detector implementation.
- Duration-plausibility guard added alongside the amplitude guard, excluding segments shorter than
  a physiologically plausible blink duration (~100 ms). Applied sequentially after the amplitude
  guard, since the two test independent failure modes: one restEO candidate segment (17 samples,
  ~34 ms) passed the amplitude check despite being far too brief to be a genuine blink.

Corrected bug (authors' code):
- Segment-padding first-branch used a hardcoded `Atrl[0,1]` reference instead of `Atrl[i,1]`, 
  incorrectly always referencing the first detected segment regardless of which segment was 
  being padded.