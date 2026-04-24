import sys
import os
sys.path.append(os.path.abspath('.'))

import numpy as np
from ecg_processing.load_ecg import load_record
from ecg_processing.preprocess import preprocess
from ecg_processing.r_peak_detection import (
    detect_rpeaks, get_rr_intervals, get_heart_rate, get_hrv_metrics
)
from ecg_processing.features import extract_features, extract_segments


def test_features(record_name='100'):
    # ── Load and preprocess ───────────────────────────────────────
    print(f"Loading record {record_name}...")
    signal, fs, labels = load_record(f'data/raw/mitdb/{record_name}')
    ten_sec = int(10 * fs)
    raw     = signal[:ten_sec, 0]
    clean   = preprocess(raw, fs)

    # ── R-peaks ───────────────────────────────────────────────────
    r_peaks      = detect_rpeaks(clean, fs)
    rr_intervals = get_rr_intervals(r_peaks, fs)
    heart_rate   = get_heart_rate(rr_intervals)
    hrv          = get_hrv_metrics(rr_intervals)

    # ── Features ──────────────────────────────────────────────────
    features = extract_features(r_peaks, rr_intervals, heart_rate, hrv)
    segments, valid_peaks = extract_segments(clean, r_peaks)

    print("\n--- Features ---")
    for k, v in features.items():
        print(f"  {k}: {v}")

    print(f"\nSegments shape : {segments.shape}")
    print(f"Valid peaks    : {len(valid_peaks)}")

    # ── Checks ────────────────────────────────────────────────────
    print("\n--- Validation Checks ---")
    assert segments.shape[1] == 180,          "FAIL: segment length wrong"
    assert segments.shape[0] > 0,            "FAIL: no segments extracted"
    assert not np.any(np.isnan(segments)),   "FAIL: NaN in segments"
    assert 40 < features["heart_rate"] < 180,"FAIL: heart rate out of range"
    assert features["beat_count"] > 0,       "FAIL: no beats counted"

    print("All checks passed")
    return features, segments


if __name__ == '__main__':
    test_features('100')