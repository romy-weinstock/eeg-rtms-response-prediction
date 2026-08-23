##Feature extraction functions 
## Author: Romy Weinstock

# Required imports
import numpy as np
import pandas as pd
import mne
from src.preprocessing import find_repo_root
from itertools import combinations
from mne_connectivity import spectral_connectivity_epochs

## Load subject epochs function 
def load_subject_epochs(subject_id, condition, variant, data_dir): 
    """
    Load preprocessed epochs for one subject, condition, and HEOG variant.

    Parameters:
    subject_id (str): The ID of the subject, e.g. 'sub-87999321'.
    condition (str): The resting condition ('restEC' or 'restEO').
    variant (str): The HEOG correction variant ('heog_off' or 'heog_on'), used to
    select the correct derivatives folder (derivatives_<variant>).
    data_dir (str or Path): The project's data directory.

    Returns:
    epochs (mne.Epochs): The preprocessed epochs loaded from
    data_dir/derivatives_<variant>/subject_id/subject_id_condition-epo.fif.
    """
    epochs_path = data_dir / f'derivatives_{variant}' / subject_id / f'{subject_id}_{condition}-epo.fif'
    epochs = mne.read_epochs(epochs_path)
    return epochs

## Get subject QC
def get_subject_qc(subject_id, condition, variant, qc_log):
    """
    Filter the combined QC log down to one subject's row for a given condition
    and HEOG variant.

    Parameters:
    subject_id (str): The ID of the subject, e.g. 'sub-87999321'.
    condition (str): The resting condition ('restEC' or 'restEO').
    variant (str): The HEOG correction variant, matched against the 'heog_variant'
    column in qc_log.
    qc_log (pd.DataFrame): The combined QC log for the full cohort, loaded once by
    the caller (not re-read from disk on every call).

    Returns:
    qc_info (dict): QC metadata for this subject/condition/variant (n_epochs_after,
    autoreject_extreme, autoreject_consensus, autoreject_n_interpolate, etc.),
    as a single flat dict.
    """
    qc_row = qc_log[
        (qc_log['subject_id'] == subject_id) &
        (qc_log['condition'] == condition) &
        (qc_log['heog_variant'] == variant)  
    ]
    assert len(qc_row) == 1, f"expected exactly 1 QC row, got {len(qc_row)}"
    qc_info = qc_row.iloc[0].to_dict()
    return qc_info

## Compute band power function 
def compute_band_power(epochs, bands):
    """
    Compute the band's power from epochs

    Parameters:
    epochs (mne.Epochs): The preprocessed epochs 
    bands (dict): band_name: (low,high) specifying band boundaries 

    Returns:
    band_power (dict): channel_bandname_power: band_mean
    """
    spectrum = epochs.compute_psd(method='welch')
    freqs = spectrum.freqs
    power_data_array = spectrum.get_data()
    band_power = {}
    for band_name, (low, high) in bands.items():
        mask = (freqs >= low) & (freqs < high)
        band_slice = power_data_array[:, :, mask]   
        band_sum = np.sum(band_slice, axis=2)         
        band_mean = np.mean(band_sum, axis=0) * 1e12
        for channel, value in zip(spectrum.ch_names, band_mean):
            band_power[f"{channel}_{band_name}_power"] = value
    assert not any(np.isnan(v) for v in band_power.values()), "NaN found in band_power"
    return band_power


## Compute PLI function

def compute_pli(epochs, bands):
    """
    Compute PLI (phase lag index) connectivity from epochs.

    Parameters:
    epochs (mne.Epochs): The preprocessed epochs
    bands (dict): band_name: (low, high) specifying band boundaries

    Returns:
    pli_results (dict): {channel_a}_{channel_b}_{band_name}_pli: PLI value,
    one entry per upper-triangle channel pair per band.
    """
    picks = mne.pick_types(epochs.info, eeg=True)
    ch_names = [epochs.ch_names[i] for i in picks]
    pairs = list(combinations(range(len(ch_names)), 2))
    seeds = [i[0] for i in pairs]
    targets = [j[1] for j in pairs]
    indices = (seeds, targets)
    fmin = tuple(low for (low, high) in bands.values())
    fmax = tuple(high for (low, high) in bands.values())
    con = spectral_connectivity_epochs(
        data=epochs,
        method='pli',
        indices=indices,
        fmin=fmin,
        fmax=fmax,
        faverage=True,
        sfreq=epochs.info['sfreq'],
        )
    con_data = con.get_data()
    pli_results = {}
    for p, (i, j) in enumerate(pairs):
        for b, band_name in enumerate(bands):
            pli_results[f"{ch_names[i]}_{ch_names[j]}_{band_name}_pli"] = con_data[p, b]
    assert all(0 <= v <= 1 for v in pli_results.values()), "PLI value outside [0,1] found"
    assert not any(np.isnan(v) for v in pli_results.values()), "NaN found in pli_results"
    return pli_results

## Orchestrator: extract subject features

def extract_subject_features(subject_id, condition, variant, data_dir, qc_log, bands):
    """
    Run the full feature extraction pipeline for one subject, condition, and
    HEOG variant.

    Parameters:
    subject_id (str): The ID of the subject, e.g. 'sub-87999321'.
    condition (str): The resting condition ('restEC' or 'restEO').
    variant (str): The HEOG correction variant, matched against the 'heog_variant'
    column in qc_log.
    data_dir (str or Path): The project's data directory.
    qc_log (pd.DataFrame): The combined QC log for the full cohort, loaded once by
    the caller (not re-read from disk on every call).
    bands (dict): band_name: (low, high) specifying band boundaries.

    Returns:
    results (dict): a single flat dict, one row's worth of data:
        - identity columns: subject_id, condition, variant
        - status column: reason ('ok' on success, description of failure otherwise)
        - QC columns: all keys from qc_info (n_epochs_after, autoreject_extreme, etc.)
        - feature columns: all keys from compute_band_power (Fp1_delta_power, etc.)
          and all keys from compute_pli (Fp1_Fp2_delta_pli, etc.)
    On failure, QC and feature columns are absent from qc_info/band_power/pli
    (since they were never computed) - the caller should expect this row to have
    fewer keys than a successful row, or NaN-fill afterward if a fixed column set
    across all rows is needed at concatenation time.
    """
    try:
        epochs = load_subject_epochs(subject_id, condition, variant, data_dir)
        qc_info = get_subject_qc(subject_id, condition, variant, qc_log)
        band_power = compute_band_power(epochs, bands)
        pli = compute_pli(epochs, bands)

        qc_info_renamed = {
            (f"preprocessing_{k}" if k in ("status", "error") else k): v
            for k, v in qc_info.items()
            }
        results = {
            "subject_id": subject_id,
            "condition": condition,
            "variant": variant,
            "reason": "ok",
            **qc_info_renamed,
            **band_power,
            **pli,
            }
    except Exception as e:
        results = {
            "subject_id": subject_id,
            "condition": condition,
            "variant": variant,
            "reason": str(e),
        }
    return results