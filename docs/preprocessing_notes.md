# Preprocessing notes: methodology, deviations, and open items

## Preprocessing status

Pipeline order (matching authors' `dataset` class methods): `bipolarEOG -> demean -> apply_filters -> correct_EOG -> epoching -> artefact rejection`.
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
- `correct_EOG` (HEOG): artefact detection, amplitude-plausibility guard, and duration-plausibility
  guard validated (see deviations below for a detection fix required before guards could be
  applied meaningfully). Gratton regression generalised across all 26 EEG channels and all 7
  guarded restEO segments, regressed onto the VEOG-corrected data per the authors' documented
  sequential order (vertical eye movements corrected first, then horizontal). A beta-plausibility
  guard was added to exclude implausible per-(channel, segment) coefficients arising from a known
  short-window regression limitation (see deviations below).
- Epoching: 5-second, non-overlapping windows (see rationale below).
- Artefact rejection: `autoreject` adopted and benchmarked against known artefacts; EMG bandpower
  supplement built and validated as a standing audit (see below).

### restEC artefact segments found to be non-ocular

Both VEOG segments originally detected in the pilot subject's restEC (eyes-closed) recording
were found, on inspection, to be step/pop artefacts rather than genuine blinks: peak-to-peak
amplitude of 11,404.8 µV and 10,234.7 µV respectively, roughly one to two orders of magnitude
above a physiologically plausible blink (typically tens to a few hundred µV). Confirmed present in
the pre-`apply_filters` data, ruling out filtering as the cause. One of these segments coincides
with the F3 discontinuity noted below near the recording's end; the other coincides with two of
the three F3 discontinuities near samples ~14,000-16,500. Both restEC segments were excluded
by the amplitude-plausibility guard. Since restEC therefore has zero valid segments for VEOG,
correction was not applied - explicitly logged rather than left implicit, matching the authors'
own pattern of recording 0-artefact cases.

The subject's restEO (eyes-open) recording was used instead for the regression proof-of-concept,
since eyes-open resting state reliably contains genuine blinks: 8 candidate segments were
detected, of which 7 passed the amplitude guard (peak-to-peak 67.9-576.6 µV) and 1 was
excluded (18,162.1 µV, again near the recording's end).

### HEOG detection: z-scoring failure and fix

Standard mean/std z-scoring (matching the authors' unmodified method) detected only 1 candidate
segment for the entire recording - a non-ocular artefact at the very end of the recording inflated
the recording-wide standard deviation by 100.25x, suppressing detection of genuine saccadic
activity visible elsewhere in the raw signal (confirmed via a rescaled amplitude plot). The same
check run on VEOG found a smaller inflation (15.11x) that did not suppress detection, since VEOG's
genuine blinks were large enough to clear the inflated threshold anyway; VEOG's method was left
unchanged. A robust alternative (median/MAD in place of mean/std) was tried and rejected: it
flagged 43.6% of the recording as artefact, since the amplitude envelope's non-negative,
right-skewed distribution collapses the median absolute deviation near zero, making the detection
threshold hypersensitive rather than robust. The adopted fix - trimmed mean/std, excluding the top
2% of envelope values from the reference statistics only - resolved this, detecting 8 plausible
candidate segments (5.72% of the recording flagged), 5 of which closely align in timing with
already-validated VEOG blink segments, independent evidence the fix detects genuine signal.

## Epoching and artefact rejection (post-EOG-correction)

**Epoch length**: 5 seconds, non-overlapping - a deliberate deviation from the authors' default
(`trllength=2`, undocumented rationale in their code). Chosen for connectivity/synchrony-estimate
quality (literature: >4s more accurate, >6s optimal) and because this project's ML pipeline
classifies at the subject level (epoch-level features aggregated per subject), so epoch count
doesn't set training-sample size the way it would for epoch-level classification.

**Artefact rejection strategy**: `autoreject` (Jas, Engemann, Bekhti, Raimondo & Gramfort, 2017,
*NeuroImage*, https://doi.org/10.1016/j.neuroimage.2017.06.030) adopted in place of replicating
the authors' seven `detect_*` methods. The authors' methods trace to their own established
clinical pipeline predating `autoreject` (TDBRAIN is a 20-year archive) - a legacy-continuity
choice, not evidence against the tool. Full replication would require guard/validation work at
the same scale as this project's VEOG/HEOG work, seven times over.

**Benchmark result**: `autoreject`, using per-channel amplitude thresholds with no knowledge of
this project's own detection/guard work, independently flagged the same artefacts found by hand -
the end-of-recording artefact (both conditions) and the Day 5 F3 discontinuities (interpolated in
restEC epochs 4-7, directly overlapping the visually-identified location). Strong convergent
validation across three independent methods.

**Documented gap and resolution**: `autoreject` operates on amplitude only, not frequency content,
whereas the authors' `detect_emg` isolates 75-95 Hz specifically. A targeted bandpower supplement
(same band/threshold as `detect_emg`, channel-within-epoch granularity) was built and cross-checked
against `autoreject`'s own flags: 0 of 26 flagged pairs were missed by `autoreject` for this
subject - the gap did not materialise here, though this is evidence for one subject, not a general
finding. Retained as a standing per-subject audit when scaling to the full cohort, rather than
treated as closed.

**Output**: final corrected, epoched, artefact-rejected data saved to
`data/derivatives/<subject_id>/` (`-epo.fif` per condition; EMG audit as a sparse per-flagged-pair
CSV, including whether each flag was missed by `autoreject`, for later cohort-level aggregation).

**Known open items**:
- Segments near the end of the recording appear artefact-prone in both restEC and restEO for
  this subject (possibly equipment settling or cap removal); not yet investigated, noted as a
  pattern to watch across subjects.
- HEOG's duration-plausibility threshold (100 ms) was reused from VEOG without independent
  validation for this subject; all 7 amplitude-retained segments happened to be well above this
  threshold (162-482 ms), so the reuse was never actually tested against a borderline case. A
  future subject with a genuinely brief HEOG artefact could pass this guard undetected.
- Short-window (per-segment) least-squares regression can produce implausibly large coefficients
  when a regressor and target happen to have similarly-shaped deflections within a short window,
  even when both signals are genuine (confirmed via direct visual inspection for one HEOG segment).
  A beta-plausibility guard (see deviations below) mitigates this but does not resolve the
  underlying limitation of segment-scoped regression on short windows.
- Gratton regression vs. ICA-based correction was considered as an alternative given known
  limitations of regression (subtracting genuine EEG signal correlated with EOG) versus ICA's
  generally stronger blink-removal performance. Decision: retained Gratton, for comparability with
  TDBRAIN's own published methodology and given the pipeline's existing validated state; not
  empirically compared head-to-head on this dataset. Documented as a deliberate trade-off, not an
  oversight.
- Open design decision: how subjects with zero valid (post-guard) segments in a given condition
  should be represented downstream at scale (currently handled via an explicit per-subject log
  for the pilot subject's restEC condition; not yet a structured, reusable format for 163 subjects).

Documented deviations from authors' code (library compatibility, no behaviour change):
- `np.int` -> `int()` (deprecated in current NumPy)
- `scipy.signal.boxcar` -> `scipy.signal.windows.boxcar` (relocated in current SciPy)

Documented deviations from authors' code (behaviour change, justified):
- One-sided z-score threshold (`z > threshold`) used in VEOG and HEOG artefact detection, in place
  of the authors' two-sided condition, since the input (`boxdata`, an amplitude envelope) is
  non-negative by construction; the two-sided condition false-flagged ~76% of the recording as
  artefact due to distribution skew.
- Amplitude-plausibility guard added before the Gratton regression (VEOG and HEOG), excluding
  segments with implausible peak-to-peak amplitude. The authors' own pipeline
  (`autopreprocess_pipeline.py`) runs `correct_EOG` before `detect_jumps`, with no equivalent
  safeguard - meaning step/pop artefacts would be regressed against as if ocular in the original
  published method. This guard is a stopgap addressing that specific gap, not a claim of
  perfectly separating artefact types, pending a proper jump-detector implementation. HEOG reuses
  VEOG's threshold value (1000 µV), justified by an equivalently clean amplitude gap in the
  observed data (7 genuine segments at 45-119 µV vs. one artefact at 11,807.5 µV).
- Duration-plausibility guard added alongside the amplitude guard (VEOG and HEOG), excluding
  segments shorter than a physiologically plausible blink duration (~100 ms). Applied sequentially
  after the amplitude guard, since the two test independent failure modes: one restEO candidate
  segment (17 samples, ~34 ms) passed the amplitude check despite being far too brief to be a
  genuine blink.
- HEOG z-scoring computed on trimmed reference statistics (top 2% of the amplitude envelope
  excluded) rather than the full recording, to prevent a single extreme non-ocular artefact from
  suppressing detection of genuine saccadic activity elsewhere. Percentile-based rather than a
  fixed µV cutoff, so it generalises across subjects without requiring prior knowledge of an
  artefact's location or amplitude.
- Beta-plausibility guard (`beta_plausibility_bound = 1.0`) added for HEOG regression, excluding
  individual (channel, segment) correction coefficients that fall outside the range seen elsewhere -
  addresses a limitation of short-window least-squares regression, not addressed anywhere in the
  authors' original method.

Corrected bug (authors' code):
- Segment-padding first-branch used a hardcoded `Atrl[0,1]` reference instead of `Atrl[i,1]`,
  incorrectly always referencing the first detected segment regardless of which segment was
  being padded.

