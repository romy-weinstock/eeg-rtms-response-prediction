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

## Function to compute coherence and PLV 
def compute_coherence_plv(epochs, bands):
    """
    Compute PLV and coherence connectivity measures from epochs.

    Parameters:
    epochs (mne.Epochs): The preprocessed epochs
    bands (dict): band_name: (low, high) specifying band boundaries

    Returns:
    coherence_plv_results (dict): {channel_a}_{channel_b}_{band_name}_{metric}:
    value, one entry per upper-triangle channel pair per band per metric
    (coh, plv).
    """
    picks = mne.pick_types(epochs.info, eeg=True)
    ch_names = [epochs.ch_names[i] for i in picks]
    pairs = list(combinations(range(len(ch_names)), 2))
    seeds = [i[0] for i in pairs]
    targets = [j[1] for j in pairs]
    indices = (seeds, targets)
    fmin = tuple(low for (low, high) in bands.values())
    fmax = tuple(high for (low, high) in bands.values())
    con_multi = spectral_connectivity_epochs(
        data=epochs,
        method=['coh', 'plv'],
        indices=indices,
        fmin=fmin,
        fmax=fmax,
        faverage=True,
        sfreq=epochs.info['sfreq'],
        )
    coherence_plv_results = {}
    for con in con_multi:
        metric_name = con.method
        con_data = con.get_data()
        for p, (i, j) in enumerate(pairs):
            for b, band_name in enumerate(bands):
                key = f"{ch_names[i]}_{ch_names[j]}_{band_name}_{metric_name}"
                coherence_plv_results[key] = con_data[p, b]
    assert not any(np.isnan(v) for v in coherence_plv_results.values()), "NaN found in results"
    assert all(0 <= v <= 1 for v in coherence_plv_results.values()), "Value outside [0,1] found"
    return coherence_plv_results

## Kuramoto function 
def compute_kuramoto(epochs, bands):
    """
    Compute Kuramoto order parameter and metastability from epochs.

    Parameters:
    epochs (mne.Epochs): The preprocessed epochs
    bands (dict): band_name: (low, high) specifying band boundaries

    Returns:
    kuramoto_results (dict): {band_name}_order: order parameter,
    {band_name}_metastability: metastability, one pair of entries per band.
    """
    picks = mne.pick_types(epochs.info, eeg=True)
    kuramoto_results = {}

    for band_name, (low, high) in bands.items():
        epochs_filt = epochs.copy().filter(l_freq=low, h_freq=high, picks=picks, verbose=False)
        epochs_filt.apply_hilbert(picks=picks, envelope=False)
        analytic = epochs_filt.get_data(picks=picks)  # (n_epochs, n_channels, n_times), complex
        theta = np.angle(analytic)

        order_per_epoch = []
        metastability_per_epoch = []
        for ep in range(theta.shape[0]):
            R_t = np.abs(np.mean(np.exp(1j * theta[ep]), axis=0))
            order_per_epoch.append(R_t.mean())
            metastability_per_epoch.append(R_t.std())

        kuramoto_results[f"{band_name}_order"] = np.mean(order_per_epoch)
        kuramoto_results[f"{band_name}_metastability"] = np.mean(metastability_per_epoch)

    assert not any(np.isnan(v) for v in kuramoto_results.values()), "NaN found in kuramoto_results"
    order_keys = [k for k in kuramoto_results if k.endswith('_order')]
    assert all(0 <= kuramoto_results[k] <= 1 for k in order_keys), "Order parameter outside [0,1] found"
    return kuramoto_results

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
        - feature columns: all keys from compute_band_power (Fp1_delta_power, etc.),
          compute_pli (Fp1_Fp2_delta_pli, etc.), compute_coherence_plv
          (Fp1_Fp2_delta_coh, Fp1_Fp2_delta_plv, etc.), and compute_kuramoto
          (delta_order, delta_metastability, etc.)
    On failure, QC and feature columns are absent from qc_info/band_power/pli/
    coherence_plv/kuramoto (since they were never computed) - the caller should
    expect this row to have fewer keys than a successful row, or NaN-fill
    afterward if a fixed column set across all rows is needed at concatenation
    time.
    """
    try:
        epochs = load_subject_epochs(subject_id, condition, variant, data_dir)
        qc_info = get_subject_qc(subject_id, condition, variant, qc_log)
        band_power = compute_band_power(epochs, bands)
        pli = compute_pli(epochs, bands)
        coherence_plv = compute_coherence_plv(epochs, bands)
        kuramoto = compute_kuramoto(epochs, bands)

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
            **coherence_plv,
            **kuramoto,
            }
    except Exception as e:
        results = {
            "subject_id": subject_id,
            "condition": condition,
            "variant": variant,
            "reason": str(e),
        }
    return results


## Compute IAF function (not integrated in the orchestrator - to run only on protocol 1 subjects)
def compute_iaf(epochs, alpha_window=(7, 13), channel='F3'):
    """
    Compute individual alpha frequency (IAF) and IAF-prox from epochs.

    Parameters:
    epochs (mne.Epochs): The preprocessed epochs
    alpha_window (tuple): (low, high) frequency bounds for peak-picking,
    following Roelofs et al. (2021)'s 7-13 Hz window - deliberately
    distinct from this project's primary 8-13 Hz alpha band (Decision 2),
    used here for replication fidelity.
    channel (str): single electrode to compute IAF at, matched to the
    10 Hz left-DLPFC stimulation site (F3), following Roelofs et al. (2021).

    Returns:
    iaf_results (dict): {'iaf': peak frequency in Hz,
    'iaf_prox': absolute distance from 10 Hz}
    """
    spectrum = epochs.compute_psd(method='welch', verbose=False)
    freqs = spectrum.freqs
    psd_data = spectrum.get_data()

    ch_idx = spectrum.ch_names.index(channel)
    low, high = alpha_window
    alpha_mask = (freqs >= low) & (freqs <= high)

    psd_ch = psd_data[:, ch_idx, :].mean(axis=0)
    psd_ch_alpha = psd_ch[alpha_mask]
    freqs_alpha = freqs[alpha_mask]

    iaf = freqs_alpha[np.argmax(psd_ch_alpha)]
    iaf_prox = abs(iaf - 10)

    iaf_results = {"iaf": iaf, "iaf_prox": iaf_prox}
    assert not any(np.isnan(v) for v in iaf_results.values()), "NaN found in iaf_results"
    assert alpha_window[0] <= iaf_results["iaf"] <= alpha_window[1], "IAF outside search window"
    return iaf_results