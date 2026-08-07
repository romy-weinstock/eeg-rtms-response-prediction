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




