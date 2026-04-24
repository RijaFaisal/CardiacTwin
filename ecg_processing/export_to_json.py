import numpy as np
import json
import os


def downsample(signal, target_points=1000):
    """
    Reduce signal to target_points for frontend display.
    Full 650k sample array is too large to send over API.
    """
    factor = max(1, len(signal) // target_points)
    return signal[::factor].tolist()


def export_to_json(record_id, signal, fs, r_peaks, rr_intervals,
                   heart_rate, hrv, features, save=True):
    """
    Build the JSON schema consumed by the frontend and API.
    This schema must never change once the frontend uses it.
    """
    r_peak_times = (r_peaks / fs).tolist()

    output = {
        "record_id":        str(record_id),
        "dataset":          "mitbih",
        "fs":               int(fs),
        "duration_seconds": round(len(signal) / fs, 2),
        "heart_rate":       features["heart_rate"],
        "beat_count":       features["beat_count"],
        "mean_rr":          features["mean_rr"],
        "std_rr":           features["std_rr"],
        "min_rr":           features["min_rr"],
        "max_rr":           features["max_rr"],
        "hrv_sdnn":         features["sdnn"],
        "hrv_rmssd":        features["rmssd"],
        "r_peak_indices":   r_peaks.tolist(),
        "r_peak_times_sec": r_peak_times,
        "rr_intervals_sec": rr_intervals.tolist(),
        "signal":           downsample(signal, target_points=1000),
    }

    if save:
        os.makedirs("data/processed", exist_ok=True)
        path = f"data/processed/{record_id}.json"
        with open(path, "w") as f:
            json.dump(output, f, indent=2)
        print(f"Saved to {path}")

    return output