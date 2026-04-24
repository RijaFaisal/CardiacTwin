import numpy as np


def extract_features(r_peaks, rr_intervals, heart_rate, hrv):
    """
    Compile all scalar features into one dictionary.
    Used for JSON export and ML feature engineering.
    """
    return {
        "heart_rate": round(float(heart_rate), 2),
        "mean_rr":    round(float(np.mean(rr_intervals)), 4),
        "std_rr":     round(float(np.std(rr_intervals)), 4),
        "rmssd":      round(float(hrv["rmssd"]), 4),
        "sdnn":       round(float(hrv["sdnn"]), 4),
        "beat_count": int(len(r_peaks)),
        "min_rr":     round(float(np.min(rr_intervals)), 4),
        "max_rr":     round(float(np.max(rr_intervals)), 4),
    }


def extract_segments(signal, r_peaks, window=180):
    """
    Cut a fixed-length window centred on each R-peak.
    Output shape: (num_beats, window) — fed directly into the CNN.
    Skips peaks too close to the signal edges.
    """
    segments = []
    valid_peaks = []
    half = window // 2

    for peak in r_peaks:
        start = peak - half
        end   = peak + half
        if start >= 0 and end < len(signal):
            segments.append(signal[start:end])
            valid_peaks.append(peak)

    return np.array(segments), np.array(valid_peaks)