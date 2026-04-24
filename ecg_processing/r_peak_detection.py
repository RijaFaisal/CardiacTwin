import numpy as np
import neurokit2 as nk


def detect_rpeaks(signal, fs):
    """
    Detect R-peak positions in a cleaned ECG signal.
    Uses neurokit2 which implements a robust Pan-Tompkins style detector.
    Input:  clean normalised signal + sampling frequency
    Output: array of sample indices where each R-peak occurs
    """
    # neurokit2 expects a 1D numpy array
    signals, info = nk.ecg_process(signal, sampling_rate=fs)
    r_peaks = info["ECG_R_Peaks"]
    return r_peaks


def get_rr_intervals(r_peaks, fs):
    """
    Compute RR intervals in seconds from R-peak sample indices.
    RR interval = time between two consecutive heartbeats.
    """
    rr_samples = np.diff(r_peaks)       # difference in samples
    rr_seconds = rr_samples / fs        # convert to seconds
    return rr_seconds


def get_heart_rate(rr_intervals):
    """
    Compute average heart rate in BPM from RR intervals.
    """
    return 60.0 / np.mean(rr_intervals)


def get_hrv_metrics(rr_intervals):
    """
    Compute basic Heart Rate Variability metrics.
    SDNN  — standard deviation of RR intervals
    RMSSD — root mean square of successive differences
    Both are standard HRV measures used in cardiology.
    """
    sdnn  = np.std(rr_intervals)
    rmssd = np.sqrt(np.mean(np.diff(rr_intervals) ** 2))
    return {"sdnn": round(sdnn, 4), "rmssd": round(rmssd, 4)}