import sys
import os
sys.path.append(os.path.abspath('.'))

import json
from ecg_processing.load_ecg import load_record
from ecg_processing.preprocess import preprocess
from ecg_processing.r_peak_detection import (
    detect_rpeaks, get_rr_intervals, get_heart_rate, get_hrv_metrics
)
from ecg_processing.features import extract_features, extract_segments
from ecg_processing.export_to_json import export_to_json


def test_export(record_name='100'):
    # ── Full pipeline ─────────────────────────────────────────────
    signal, fs, labels = load_record(f'data/raw/mitdb/{record_name}')
    ten_sec      = int(10 * fs)
    raw          = signal[:ten_sec, 0]
    clean        = preprocess(raw, fs)
    r_peaks      = detect_rpeaks(clean, fs)
    rr_intervals = get_rr_intervals(r_peaks, fs)
    heart_rate   = get_heart_rate(rr_intervals)
    hrv          = get_hrv_metrics(rr_intervals)
    features     = extract_features(r_peaks, rr_intervals, heart_rate, hrv)

    # ── Export ────────────────────────────────────────────────────
    result = export_to_json(
        record_id    = record_name,
        signal       = clean,
        fs           = fs,
        r_peaks      = r_peaks,
        rr_intervals = rr_intervals,
        heart_rate   = heart_rate,
        hrv          = hrv,
        features     = features,
        save         = True
    )

    # ── Print and validate ────────────────────────────────────────
    print("\n--- JSON Output ---")
    for k, v in result.items():
        if k == "signal":
            print(f"  signal: [{v[0]:.4f}, {v[1]:.4f}, ...] ({len(v)} points)")
        else:
            print(f"  {k}: {v}")

    print("\n--- Validation Checks ---")
    assert result["record_id"] == record_name,    "FAIL: record_id wrong"
    assert result["fs"] == int(fs),               "FAIL: fs wrong"
    assert len(result["signal"]) <= 2000,         "FAIL: signal not downsampled"
    assert len(result["r_peak_indices"]) > 0,     "FAIL: no R-peaks in JSON"
    assert len(result["r_peak_times_sec"]) > 0,   "FAIL: no peak times in JSON"
    assert 40 < result["heart_rate"] < 180,       "FAIL: heart rate out of range"

    # Confirm file was saved
    path = f"data/processed/{record_name}.json"
    assert os.path.exists(path), "FAIL: JSON file not saved"

    print("All checks passed")
    print(f"\nJSON saved to data/processed/{record_name}.json")


if __name__ == '__main__':
    test_export('100')