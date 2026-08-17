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
-`correct_EOG` (HEOG): artefact detection and guards validated on the pilot subject (see
  deviations below for the initial detection fix). **Since substantially revised** - see
  "Small-batch stress test" and "HEOG baseline-drift removal" sections below: duration bounds
  were tightened, then baseline-drift removal was added and validated via a window-size sweep.
  HEOG correction confidence is meaningfully improved but remains not fully resolved -
  `apply_heog_correction` defaults to `False`.
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

### VEOG z-scoring: extending the trimmed-mean fix

The reusable batch module (`src/preprocessing.py`) applies trimmed mean/std z-scoring to VEOG
as well as HEOG, rather than restricting the fix to HEOG only (as in the pilot notebook). This
is a deliberate deviation from the pilot notebook's original treatment, not an oversight.

**Rationale**: trimming the top 2% of the amplitude envelope before computing reference
mean/std costs little when a channel has no inflation problem (the excluded tail is small and
the trimmed statistics converge close to the untrimmed ones), but meaningfully protects against
the case where one extreme artefact skews detection sensitivity elsewhere in the recording - the
exact failure mode found for HEOG. Given this asymmetry, trimmed z-scoring was adopted as the
single default for both channels rather than a per-channel choice.

**Verified consequence, pilot subject (restEC)**: VEOG detection under trimming found 4
candidate segments versus the pilot notebook's 2 (under standard z-scoring). The two segments
matching the notebook's original findings are unchanged (both later excluded by the
amplitude-plausibility guard as non-ocular step artefacts, consistent with the notebook). The
two additional segments (`[10129:10257]`, `[20870:21212]`) were visually inspected and confirmed
as genuine blinks - smooth, symmetric rise/fall shape, peak amplitudes ~150-165 µV, within the
<300-400 µV range typical for genuine blinks at Fp1/VEOG (see amplitude-guard rationale above) -
and both pass the amplitude and duration guards. This suggests standard z-scoring under-detected
real blinks on this subject, for the same reason it under-detected HEOG saccades: the same
non-ocular artefacts inflating the reference standard deviation and raising the effective
detection threshold.

**Open item**: verified on one subject only. Whether trimmed z-scoring on VEOG is uniformly
beneficial (versus occasionally introducing false-positive detections) across the cohort is
untested - a specific target for the small-batch stress test before full-cohort scaling.

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
- `detect_artefact_segments`'s parameters - `blink_amplitude_threshold_uv` (1000 µV) and
  `trim_pct` (2%) - were both set from pilot-subject-specific observations (this subject's step
  artefacts and end-of-recording artefact's magnitude/extent). Sorted as empirically-fit rather
  than physiologically-grounded; require stress-testing on a small batch of additional subjects,
  particularly ones with noisier or differently-shaped artefact profiles, before full-cohort use.

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
- Trimmed z-scoring (top 2% excluded) applied to VEOG as well as HEOG in the reusable batch
  module, extending a fix originally scoped to HEOG only in the pilot notebook. See "VEOG
  z-scoring" section above for rationale and pilot-subject verification.

Corrected bug (authors' code):
- Segment-padding first-branch used a hardcoded `Atrl[0,1]` reference instead of `Atrl[i,1]`,
  incorrectly always referencing the first detected segment regardless of which segment was
  being padded.

## Module refactor: full pipeline in src/preprocessing.py (complete)

All pipeline steps validated in this document have been refactored into reusable functions
in `src/preprocessing.py`, each independently validated against the pilot notebook's results
before being treated as correct. Twelve functions plus one orchestrator:

`load_and_prepare_raw`, `bipolarEOG`, `demean`, `apply_filters`, `detect_artefact_segments`,
`amplitude_guard`, `duration_guard`, `gratton_regression`, `epoch_raw`, `run_autoreject`,
`emg_bandpower_audit`, `save_epochs`, and `preprocess_subject` (orchestrator: runs the full
chain for one subject, both conditions, with per-condition error handling).

## Small-batch stress test: findings and fixes

Ran `preprocess_subject` on 6 additional subjects (stratified by age tercile,
responder/non-responder balance, and self-reported sleep/wellness as a noise-risk proxy),
plus the pilot subject - 14 subject/condition runs total, 0 errors.

**`autoreject` parameter instability**: 5 of 12 initial runs showed `consensus=1.00,
n_interpolate~25` (near-maximal channel count) - an extreme, minimally-discriminating
parameter combination. Investigated via `cv=5` (vs. default 10): did not resolve the pattern,
and destabilised a previously-normal result in the control subject, ruling out fold count as
the primary cause. Follow-up random-seed test (`sub-88030641`, `cv=10`, seeds 0/7/42) found
the instability is condition-specific: restEC gave the identical extreme result across all
three seeds (stable, not noise); restEO varied meaningfully with seed. A majority-vote
epoch-drop ensemble was built and tested (see `03_batch_test.ipynb`) but not adopted: it only
addresses epoch-drop decisions, not per-channel interpolation quality, which matters more for
this project's subject-level, epoch-aggregated features. Decision: retain `autoreject`'s
default parameters (`cv=10`, fixed `random_state`) for reproducibility, and instead capture
`consensus`/`n_interpolate` as QC metadata per subject/condition
(`autoreject_consensus`, `autoreject_n_interpolate`, `autoreject_extreme` in
`preprocess_subject`'s output), flagging `consensus >= 0.9`. Planned use: a sensitivity check
at the modelling stage (does classification performance change if flagged subjects are
excluded or down-weighted), rather than a preprocessing-stage fix for an incompletely
characterised problem.

**HEOG detection over-flagging drift as valid segments**: batch HEOG candidate counts
(30-86 per subject) were far higher than the pilot subject's 2-8. Visual inspection of 6
sampled "valid" segments (`sub-88047245`, restEC) found durations from 92 ms to 4,112 ms -
several multiples longer than any genuine saccade. Literature confirms saccades are ballistic
movements typically lasting 30-120 ms (patent US5726916), with mean durations ~36-37 ms in a
conjunctive visual search task (SD ~14 ms; Resca, Greenwood & Keech, 2013, in *Eye Movement*,
ed. L. C. Stewart, Nova Science Publishers, ISBN 978-1-62808-601-0), and that saccades and
baseline EOG drift are recognised as occupying distinct timescales (Bögels & Kayser, 2022,
*Journal of Neurophysiology*, https://doi.org/10.1152/jn.00076.2022). `duration_guard`
previously enforced only a minimum duration; added an optional `max_duration_threshold`
parameter (defaults to `None`, so VEOG calls are unaffected), applied at 200 ms for HEOG only
in `preprocess_subject` (new parameter `heog_max_duration_threshold`). Confirmed effective:
`sub-88047245` restEC dropped from 53 to 7 valid candidates, all previously-excluded segments
exceeding 200 ms.

**HEOG correction confidence - open item, not resolved**: a follow-up visual check of the 7
post-fix survivors found the improvement is real but incomplete. Unlike VEOG's validated blinks
(clean, isolated deflection returning to a visibly quiet baseline), most of the 7 HEOG segments
showed step-shifts to a new baseline level, continuous trends with no distinct feature at the
flagged region, or a deflection against an already-noisy baseline. Two explanations remain
undistinguished: shorter-duration noise/drift still passing the guard, or HEOG's baseline being
inherently noisier than VEOG's by nature (continuous small eye-position adjustments vs.
discrete blinks). HEOG-based correction should be treated as lower-confidence than VEOG-based
correction pending further investigation.

**Would `autoreject` catch mis-corrected drift downstream, as a safety net?** No - confirmed
not reliable. `autoreject` only operates on EEG-typed channels; it never sees HEOG directly.
The indirect risk (a drift-derived beta imprinting a spurious trend onto EEG channels via
Gratton regression) is only caught if the resulting distortion happens to exceed `autoreject`'s
amplitude thresholds, which is not guaranteed for a smooth, low-amplitude trend. This
confirmed the guard-level fix was necessary rather than optional.

**Re-verification after the fix**: re-ran the full 7-subject batch through `preprocess_subject`
with `heog_max_duration_threshold=200` wired in. All 14 subject/condition runs succeeded (0
errors); epoch counts after `autoreject` were consistent with the pre-fix batch run for 6 of 7
subjects. One exception: `sub-88052329` restEO dropped 4 epochs post-fix versus 1 pre-fix, with
no other parameter changed between runs - plausibly explained by the HEOG fix altering which
segments were corrected, slightly changing the EEG signal `autoreject` evaluates, but not
confirmed. Noted as an observed difference, not investigated further.

## HEOG baseline-drift removal: validated and integrated

**Summary of what was done**: initial duration-bound tightening (100-200ms) reduced HEOG
candidates but most survivors still showed step-shift baseline behaviour rather than isolated
events. Literature review confirmed this is expected physiology for genuine saccades (baseline
relocates with gaze position), not necessarily noise - but the detection method's z-score
threshold, computed against a recording-wide reference, cannot distinguish "baseline moved" from
"genuine transient event." Two detrending approaches were tested to address this before
detection: polynomial fitting (failed - dominated by outlier artefacts even with clipping,
stayed flat against visible drift) and rolling-median baseline subtraction (worked - visually
confirmed to track baseline wobble and level shifts). A window-size sweep (0.5-2.0s) found 1.0s
best: shorter windows introduced new spurious detections, longer windows barely differed from no
detrending. At 1.0s, visual inspection of all 13 surviving segments found roughly half showing
the expected isolated-deflection signature, versus ~1 in 9 without detrending.

**Integrated into the pipeline**: `remove_baseline_drift` (rolling median, `window_sec=1.0`) is
now applied to HEOG before artefact detection in `preprocess_subject`
(`heog_baseline_removal=True` by default). VEOG is unaffected.

**Still open**: roughly half of post-detrending segments show trend/step character rather than a
clean isolated event. `apply_heog_correction` remains `False` by default - detection is
meaningfully improved but not yet reliable enough to trust for correction. A known inconsistency
is also flagged in code: if `apply_heog_correction` is ever enabled, `gratton_regression`
currently regresses against the original (non-detrended) HEOG signal, not the detrended version
used for detection - unresolved, documented in the function's docstring.

**Verification below**: re-running the orchestrator on `sub-88047245` to confirm the updated
`preprocess_subject` reflects this integration correctly (expect `heog_n_candidates`/
`heog_n_valid` matching the 1.0s sweep result: 73/13 for restEC).

**Updated open items** :
- HEOG correction confidence: baseline-drift removal integrated and validated via window-size
  sweep, improving detection meaningfully (see above). Roughly half of segments still show
  step/trend character rather than isolated events - `apply_heog_correction` remains `False`
  pending further work.
- Known inconsistency (documented in code): if HEOG correction is ever enabled, regression
  would run against non-detrended HEOG data, not the detrended version used for detection -
  unresolved.
- `blink_amplitude_threshold_uv=1000` and `trim_pct=2` (VEOG/general) remain
  empirically-fit-to-pilot-subject in origin, exercised across 7 subjects without evidence of
  failure.
- Orchestrator's per-condition `try/except` has never been triggered by a real failure.
- Full cohort (163 subjects) not yet run - only 7 tested to date.