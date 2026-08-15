##Preprocessing functions for EEG data 
## Author: Romy Weinstock

#Required imports
import numpy as np
import pandas as pd 
from pathlib import Path    
import mne
from scipy.stats import zscore   
import scipy.signal

#Find the root of the repository to set the data directory

def find_repo_root(marker = ".git"): 
    """
    Find the root directory of the repository by looking for a marker file or directory (default is .git).

    """
    current_path = Path.cwd()
    for parent in [current_path, *current_path.parents]:
        if (parent / marker).exists():
            return parent
    raise FileNotFoundError(f"Could not find {marker} in any parent directory of {current_path}")
 

##Load raw BDF + fix channel types function
def load_and_prepare_raw(subject_id, condition, data_dir):
    """
    Load raw EEG data from a BDF file, set channel types, and return the raw object.

    Parameters:
    subject_id (str): The ID of the subject.
    condition (str): The experimental condition.
    data_dir (str or Path): The directory where the BDF files are stored.

    Returns:
    raw (mne.io.Raw): The loaded and prepared raw EEG data.
    """
    # Construct the file path
    subject_dir = data_dir / "TDBRAIN_Dataset_V3_1" / subject_id / "ses-1" / "eeg"
    condition_file = subject_dir / f"{subject_id}_ses-1_task-{condition}_eeg.bdf"
    #Existence check for the file
    if not condition_file.exists():
        raise FileNotFoundError(f"The file {condition_file} does not exist.")
    # Load the raw data
    condition_raw = mne.io.read_raw_bdf(condition_file, preload=True)
    # Set channel types 
    new_types = {
    'VPVA': 'eog',
    'VNVB': 'eog',
    'HPHL': 'eog',
    'HNHR': 'eog',
    'Erbs': 'ecg',
    'Mass': 'emg'
    }
    #apply the new channel types
    condition_raw.set_channel_types(new_types)
    return condition_raw


#'bipolarEOG' function  
def bipolarEOG(raw):
    """
    Create a bipolar EOG channel by subtracting the vertical EOG channels.

    Parameters:
    raw (mne.io.Raw): The raw EEG data.

    Returns:
    raw (mne.io.Raw): The raw EEG data with the new bipolar EOG channel added and the other EOG channels dropped.
    """
    # Check if the required channels exist
    if 'VPVA' not in raw.ch_names or 'VNVB' not in raw.ch_names or 'HPHL' not in raw.ch_names or 'HNHR' not in raw.ch_names:
        raise ValueError("Required EOG channels 'VPVA', 'VNVB', 'HPHL', and 'HNHR' are not present in the data.")
    #Extract the four raw channels' data for the bipolar EOG calculation
    eog_data = raw.get_data(picks=['VPVA', 'VNVB', 'HPHL', 'HNHR'])

    # Create a new channel for bipolar EOG
    VEOG = eog_data[0, :] - eog_data[1, :]  # Vertical EOG
    HEOG = eog_data[2, :] - eog_data[3, :]  # Horizontal EOG
    
    #Combine the derived signals into a single array
    bipolar_eog_data = np.vstack((VEOG, HEOG))
    #Build an Info object describing these two new channels
    info = mne.create_info(ch_names=['VEOG', 'HEOG'], sfreq=raw.info['sfreq'], ch_types=['eog', 'eog'])

    #Construct the RawArray from the stacked data and this Info
    bipolar_eog = mne.io.RawArray(bipolar_eog_data, info)

    #Merge it into the original Raw object
    raw.add_channels([bipolar_eog], force_update_info=True) # force_update_info: new channels' auto-generated Info has no lowpass/highpass history; keep raw's original values as source of truth

    #Drop VPVA/VNVB/HPHL/HNHR
    raw.drop_channels(['VPVA', 'VNVB', 'HPHL', 'HNHR'])
    
    return raw

##Demean function 
def demean(raw):
    """
    Demean the EEG data by subtracting the mean of each channel.

    Parameters:
    raw (mne.io.Raw): The raw EEG data.

    Returns:
    raw (mne.io.Raw): The demeaned raw EEG data.
    """
    # Identify the channels to demean (EEG, EOG, ECG)
    demean_picks = mne.pick_types(raw.info, eeg=True, eog=True, ecg=True, emg=False, stim=False)

     # use apply_function to demean the data
    raw.apply_function(lambda x: x - np.mean(x), picks=demean_picks)

    return raw

## Apply filtering to the raw data function 
def apply_filters(raw, hpfreq = 0.5, lpfreq = 100, notchfreq = 50, Q = 100):
    """
    Apply notch, high-pass, and low-pass filters to the raw EEG data.
    Parameters:
    raw (mne.io.Raw): The raw EEG data.
    hpfreq (float): High-pass filter cutoff frequency in Hz. Default is 0.5 Hz.
    lpfreq (float): Low-pass filter cutoff frequency in Hz. Default is 100 Hz.
    notchfreq (float): Notch filter frequency in Hz. Default is 50 Hz.
    Q (float): Quality factor for the notch filter. Default is 100.
    Returns:
    raw (mne.io.Raw): The filtered raw EEG data.
    """
    Fs = raw.info['sfreq']
    picks = mne.pick_types(raw.info, eeg=True, eog=True, ecg=True, emg=True, stim=False)
    def _filter_block(data, Fs, hpfreq, lpfreq, notchfreq, Q):
        b_notch, a_notch = scipy.signal.iirnotch(w0=notchfreq, Q=Q, fs=Fs)
        b_high, a_high = scipy.signal.butter(N=4, Wn=hpfreq/(Fs/2), btype='highpass')
        b_low, a_low = scipy.signal.butter(N=4, Wn=lpfreq/(Fs/2), btype='lowpass')

        data = scipy.signal.filtfilt(b_notch, a_notch, data)
        data = scipy.signal.filtfilt(b_high, a_high, data)
        data = scipy.signal.filtfilt(b_low, a_low, data)
        return data
    raw = raw.apply_function(_filter_block, picks=picks, Fs=Fs, hpfreq=hpfreq, lpfreq=lpfreq, notchfreq=notchfreq, Q=Q, channel_wise=False)
    return raw

## Detect artefacts function
def detect_artefact_segments(channel, Fs, z_threshold=0.2, padding=0.3, trim_pct=2):
    """
    Detect artefact segments in a given channel based on z-score thresholding.

    Parameters:
    channel (np.ndarray): The EEG channel data.
    Fs (float): The sampling frequency.

    z_threshold (float): The z-score threshold for detecting artefacts. Default is 0.2.
    padding (float): Fraction of each segment's own width to pad on either side. Default is 0.3 (30%).
    trim_pct (float): Percentage of highest-amplitude samples excluded when computing the reference mean/std for z-scoring, to prevent extreme-amplitude artefacts from distorting the threshold. Default is 2 (i.e., the top 2% of samples are excluded).

    Returns:
    artefact_segments (numpy ndarray): A numpy ndarray indicating the start and end indices of detected artefact segments padded.
    """
    # Preprocess the channel: apply a low-pass filter to remove high-frequency noise
    nyquist_freq = Fs / 2
    normal_cutoff = 15 / nyquist_freq
    b, a = scipy.signal.butter(4, normal_cutoff, btype='lowpass', analog=False)
    filtered_channel = scipy.signal.filtfilt(b, a, channel)

    #Get the analytical signal using the Hilbert transform
    filtered_channel_1D = filtered_channel[0] if filtered_channel.ndim > 1 else filtered_channel
    n_samples = len(filtered_channel_1D)
    n_pad = int(n_samples + n_samples * 0.2)  # Pad by 20% of the number of samples
    analytic_signal = scipy.signal.hilbert(filtered_channel_1D, N=n_pad, axis = -1)
    analytic_signal = np.abs(analytic_signal[:n_samples])  # Get the magnitude of the analytic signal
    # Smoothing amplitude envelope with boxcar convolution
    boxdata = scipy.signal.convolve(analytic_signal, scipy.signal.windows.boxcar(int(0.2 * Fs)), mode='same', method='direct')
    # Trim the signal to avoid edge effects
    cutoff = np.percentile(boxdata, 100 - trim_pct)
    trim_mask = boxdata <= cutoff
    trimmed_mean = boxdata[trim_mask].mean()
    trimmed_std = boxdata[trim_mask].std()
    zdata = (boxdata - trimmed_mean) / trimmed_std

    # Identify samples exceeding the threshold
    Asamps = np.where(zdata > z_threshold)[0]
    if len(Asamps) == 0: return np.empty((0, 2), dtype=int)
    # Collapsing flagged samples (Asamps_h) into contiguous artefact segments (start/end sample pairs)
    segments = []
    begin = Asamps[0]
    for i in range (len(Asamps)):
        if i>= len(Asamps)-1:
            end = Asamps[-1]
            segments.append((begin, end))
        elif Asamps[i+1] == Asamps[i]+1:
             continue
        else:
            end = Asamps[i]
            segments.append((begin, end))
            begin = Asamps[i+1]
    Atrl = np.array(segments, dtype=int)

    # Segment padding 
    artsamples = np.zeros(len(boxdata), dtype=int)
    if len(Atrl) > 0:
        for i in range(Atrl.shape[0]):
             # Case A: segment starts right at sample 0
            if Atrl[i, 0] == 0:
                pad_amount = int((Atrl[i, 1] - 0) * padding)
                artsamples[0:Atrl[i, 1] + pad_amount] = 1
            
            # Case B: segment ends at the very last sample of the recording
            elif Atrl[i, 1] == len(artsamples):
                pad_amount = int((Atrl[i, 1] - Atrl[i, 0]) * padding)
                artsamples[Atrl[i, 0] - pad_amount : len(artsamples)] = 1
            
            # Case C: general case - segment is somewhere in the middle, pad both sides
            else:
                pad_amount = int((Atrl[i, 1] - Atrl[i, 0]) * padding)
                artsamples[Atrl[i, 0] - pad_amount : Atrl[i, 1] + pad_amount] = 1
    #Rederive the final artefact segments from the padded sample map
    start = np.where(np.diff(artsamples) == 1)[0] + 1
    ends = np.where(np.diff(artsamples) == -1)[0] + 1
    if artsamples[-1] == 1: ends = np.append(ends, len(artsamples))  # recording ends inside a flagged region
    if artsamples[0] == 1: start = np.insert(start, 0, 0)            # recording starts inside a flagged region
    Atrl_padded = np.column_stack((start, ends))
    return Atrl_padded


## Amplitude guard to exclude segments whose peak-to-peak VEOG amplitude is implausible for a genuine blink
def amplitude_guard(channel_data, segments, threshold=1000):
    """
    Apply an amplitude guard to exclude segments whose peak-to-peak amplitude is implausible for a genuine blink.

    Parameters:
    channel_data (np.ndarray): The channel data, expected in volts (MNE's native unit), not µV.
    segments (list of tuples): A list of tuples indicating the start and end indices of segments.
    threshold (float): The amplitude threshold in microvolts. Default is 1000.

    Returns:
    valid_segments (list of tuples): A list of tuples indicating the start and end indices of valid segments.
    excluded_segments (list of tuples): A list of tuples indicating the start and end indices of excluded segments.
    """
    valid_segments, excluded_segments = [], []
    for (start, end) in segments:
        p2p = channel_data[start:end].max() - channel_data[start:end].min()
        p2p_uv = p2p * 1e6
        if p2p_uv > threshold:
            excluded_segments.append((start, end))
        else:
            valid_segments.append((start, end))
    return valid_segments, excluded_segments

## Duration guard to exclude segments that are too short to be a genuine blink
def duration_guard(segments, Fs, blink_duration_threshold = 100):
    """
    Apply a duration guard to exclude segments that are too short to be a genuine blink.

    Parameters:
    Fs (float): The sampling frequency in Hz. 
    segments (list of tuples): A list of tuples indicating the start and end indices of segments that passed the amplitude guard.
    blink_duration_threshold (float): The minimum duration in milliseconds. Default is 100 ms.

    Returns:
    valid_segments (list of tuples): A list of tuples indicating the start and end indices of valid segments.
    excluded_segments (list of tuples): A list of tuples indicating the start and end indices of excluded segments.
    """
    valid_segments, excluded_segments = [], []
    for (start, end) in segments:
        duration = (end - start) / Fs * 1000
        if duration < blink_duration_threshold:
            excluded_segments.append((start, end))
        else:
            valid_segments.append((start, end))
    return valid_segments, excluded_segments    

## Gratton regression to remove VEOG/ HEOGfrom EEG channels function
def gratton_regression(raw, eog_channel, valid_segments, beta_plausibility_bound = 1.0):
    """
    Apply Gratton regression to remove EOG artifacts from EEG channels.

    Parameters:
    raw (mne.io.Raw): The raw EEG data.
    eog_channel (str): The name of the EOG channel to use for regression.
    valid_segments (list of tuples): A list of tuples indicating the start and end indices of valid segments.
    beta_plausibility_bound (float): The plausibility bound for the regression coefficient beta. Default is 1.0.

    Returns:
    raw (mne.io.Raw): The raw EEG data with EOG artifacts removed.
    beta_df (pd.DataFrame): A DataFrame containing the regression coefficients and related information.
    """
    # Get the EOG channel data
    eog_data = raw.get_data(picks=[eog_channel])[0]
    # Get channel names for EEG channels
    eeg_channel_names = [raw.ch_names[i] for i in mne.pick_types(raw.info, eeg=True, eog=False, ecg=False, emg=False, stim=False)]
    # Initialize a list to store the regression coefficients
    beta_rows = []

    for channel in eeg_channel_names:
        # Get the EEG channel data
        ch_data = raw.get_data(picks=[channel])[0]
        # Initialize a list to store the regression coefficients for this channel
        for seg_id, (start, end) in enumerate(valid_segments):
            #slice this channel's target and regressor data for this segment
            ch_seg = ch_data[start:end]
            eog_seg = eog_data[start:end]
            # Estimate beta for this (channel, segment) pair
            a = eog_seg.reshape(-1, 1)
            b = ch_seg
            x, residuals, rank, s = np.linalg.lstsq(a, b, rcond=None)
            beta = x[0]
            # Check if beta is within the plausibility bound
            flagged = abs(beta) > beta_plausibility_bound 
            N = end - start
            if not flagged: # Tukey taper, scaled by this segment's own beta
                taper = scipy.signal.windows.tukey(N, alpha=0.025)
                eog_weight = taper * beta
                corrected_seg = ch_seg - (eog_weight * eog_seg)
                raw[channel, start:end] = corrected_seg
            # Store the beta value for this segment
            beta_rows.append({
                'channel': channel, 
                'segment_id': seg_id,
                'beta': beta,
                'start': start,
                'end': end,
                'n_samples': N,
                'flagged': flagged
            })  
        
    # Build the DataFrame once, after all channels and segments are done
    beta_df = pd.DataFrame(beta_rows, columns=['channel', 'segment_id', 'beta', 'start', 'end', 'n_samples', 'flagged'])
    return raw, beta_df

## Epoch raw data into fixed-length, non-overlapping windows
def epoch_raw(raw, epoch_duration=5.0):
    """
    Segment continuous raw data into fixed-length, non-overlapping epochs.

    Parameters:
    raw (mne.io.Raw): The corrected, filtered raw EEG data.
    epoch_duration (float): Epoch length in seconds. Default is 5.0s.

    Returns:
    epochs (mne.Epochs): Fixed-length epochs, no overlap.
    """
    epochs = mne.make_fixed_length_epochs(raw, duration=epoch_duration, overlap=0.0, preload=True)
    return epochs

## Set standard montage and run autoreject
def run_autoreject(epochs, montage_name='standard_1020', random_state=42):
    """
    Attach standard electrode positions and run autoreject on EEG channels.

    Parameters:
    epochs (mne.Epochs): Fixed-length epochs (pre-artefact-rejection).
    montage_name (str): MNE standard montage name. Default 'standard_1020'.
    random_state (int): Random seed for autoreject reproducibility. Default 42.

    Returns:
    epochs_clean (mne.Epochs): Epochs after autoreject (bad epochs dropped, bad channels interpolated).
    reject_log (autoreject.RejectLog): Per-(epoch, channel) rejection labels.
    """
    from autoreject import AutoReject

    montage = mne.channels.make_standard_montage(montage_name)
    epochs.set_montage(montage, on_missing='warn')

    picks_eeg = mne.pick_types(epochs.info, eeg=True)
    ar = AutoReject(picks=picks_eeg, random_state=random_state)
    epochs_clean, reject_log = ar.fit_transform(epochs, return_log=True)

    return epochs_clean, reject_log
    

## EMG bandpower audit: flags (epoch, channel) pairs with elevated 75-95 Hz power,
## cross-checked against autoreject's own flags to find what autoreject missed
def emg_bandpower_audit(epochs, reject_log, emg_band=(75, 95), emg_threshold=4):
    """
    Flag (epoch, channel) pairs with elevated EMG-band power, and report which of
    those were NOT already caught by autoreject.

    Parameters:
    epochs (mne.Epochs): Pre-autoreject epochs (montage already set).
    reject_log (autoreject.RejectLog): autoreject's own per-(epoch, channel) labels.
    emg_band (tuple): (low, high) Hz for the EMG bandpass. Default (75, 95).
    emg_threshold (float): SD threshold for flagging. Default 4.

    Returns:
    audit_df (pd.DataFrame): One row per (epoch, channel) pair flagged by the EMG
        check, with a column indicating whether autoreject missed it.
    """
    eeg_ch_names = [epochs.ch_names[i] for i in mne.pick_types(epochs.info, eeg=True)]

    epochs_emg_band = epochs.copy().filter(l_freq=emg_band[0], h_freq=emg_band[1],
                                             picks='eeg', verbose=False)
    data = epochs_emg_band.get_data(picks='eeg')  # (n_epochs, n_channels, n_times)
    bandpower = (data ** 2).mean(axis=2)  # mean squared amplitude per (epoch, channel)

    z = (bandpower - bandpower.mean(axis=0)) / bandpower.std(axis=0)
    emg_flags = np.abs(z) > emg_threshold

    eeg_cols = [reject_log.ch_names.index(ch) for ch in eeg_ch_names]
    ar_flagged = (reject_log.labels != 0)[:, eeg_cols]

    rows = []
    epoch_idx, ch_idx = np.where(emg_flags)
    for e, c in zip(epoch_idx, ch_idx):
        rows.append({
            'epoch': e,
            'channel': eeg_ch_names[c],
            'missed_by_autoreject': not ar_flagged[e, c]
        })
    audit_df = pd.DataFrame(rows, columns=['epoch', 'channel', 'missed_by_autoreject'])
    return audit_df

## Save preprocessed epochs to disk, MNE's native format
def save_epochs(epochs_clean, subject_id, condition, data_dir):
    """
    Save final artefact-rejected epochs to data/derivatives/<subject_id>/.

    Parameters:
    epochs_clean (mne.Epochs): Post-autoreject epochs.
    subject_id (str): Subject ID, used in output filename and folder.
    condition (str): Condition ('restEC' or 'restEO'), used in filename.
    data_dir (str or Path): Base data directory (containing 'derivatives/').

    Returns:
    output_path (Path): Path the file was saved to.
    """
    output_dir = data_dir / "derivatives" / subject_id
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{subject_id}_{condition}-epo.fif"
    epochs_clean.save(output_path, overwrite=True)
    return output_path

## Orchestrator: run the full pipeline for one subject, both conditions
def preprocess_subject(subject_id, data_dir, beta_plausibility_bound=1.0,
                        blink_amplitude_threshold_uv=1000, blink_duration_threshold=100,
                        z_threshold=0.2, trim_pct=2, epoch_duration=5.0):
    """
    Run the full preprocessing pipeline for one subject, both conditions (restEC, restEO).

    Returns:
    results (dict): keyed by condition, each value a dict with 'status', 'n_epochs_before',
        'n_epochs_after', 'emg_audit_df', 'output_path' (or 'error' if failed).
    """
    results = {}
    for condition in ['restEC', 'restEO']:
        try:
            raw = load_and_prepare_raw(subject_id, condition, data_dir)
            raw = bipolarEOG(raw)
            raw = demean(raw)
            raw = apply_filters(raw)

            Fs = raw.info['sfreq']
            for eog_channel in ['VEOG', 'HEOG']:
                ch_data = raw.get_data(picks=[eog_channel])[0]
                segments = detect_artefact_segments(ch_data, Fs, z_threshold=z_threshold, trim_pct=trim_pct)
                valid_amp, _ = amplitude_guard(ch_data, segments, threshold=blink_amplitude_threshold_uv)
                valid_dur, _ = duration_guard(valid_amp, Fs, blink_duration_threshold=blink_duration_threshold)
                raw, _ = gratton_regression(raw, eog_channel, valid_dur, beta_plausibility_bound=beta_plausibility_bound)

            epochs = epoch_raw(raw, epoch_duration=epoch_duration)
            n_before = len(epochs)
            epochs_clean, reject_log = run_autoreject(epochs)
            n_after = len(epochs_clean)

            emg_audit_df = emg_bandpower_audit(epochs, reject_log)
            emg_audit_df['subject_id'] = subject_id
            emg_audit_df['condition'] = condition

            output_path = save_epochs(epochs_clean, subject_id, condition, data_dir)

            results[condition] = {
                'status': 'ok',
                'n_epochs_before': n_before,
                'n_epochs_after': n_after,
                'emg_audit_df': emg_audit_df,
                'output_path': output_path
            }
        except Exception as e:
            results[condition] = {'status': 'error', 'error': str(e)}
    return results