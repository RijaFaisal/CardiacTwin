import numpy as np
from scipy.signal import butter, filtfilt, medfilt


def bandpass_filter(signal, fs, lowcut=0.5, highcut=40.0, order=4):
    """
    Remove high-frequency noise and low-frequency drift.
    Keeps the clinically relevant ECG frequency range (0.5–40 Hz).
    Uses zero-phase filtering (filtfilt) so no time shift is introduced.
    """
    nyq = fs / 2
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, signal)


def remove_baseline(signal, fs, window_ms=200):
    """
    Remove slow baseline wander caused by breathing or electrode movement.
    Uses a median filter — the window must be odd-sized.
    """
    window = int((window_ms / 1000) * fs)
    if window % 2 == 0:
        window += 1
    baseline = medfilt(signal, kernel_size=window)
    return signal - baseline


def normalise(signal):
    """
    Z-score normalisation.
    Output will have mean ~0 and std ~1.
    Required so all records are on the same scale for ML input.
    """
    mean = np.mean(signal)
    std = np.std(signal)
    if std == 0:
        raise ValueError("Signal has zero standard deviation — likely a flat/corrupt record.")
    return (signal - mean) / std


def preprocess(signal, fs):
    """
    Master function — runs all three steps in order.
    Input:  raw signal array + sampling frequency
    Output: clean, normalised signal ready for R-peak detection and ML
    """
    signal = bandpass_filter(signal, fs)
    signal = remove_baseline(signal, fs)
    signal = normalise(signal)
    return signal