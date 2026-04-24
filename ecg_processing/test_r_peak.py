import sys
import os
sys.path.append(os.path.abspath('.'))

import numpy as np
import matplotlib.pyplot as plt
from ecg_processing.load_ecg import load_record
from ecg_processing.preprocess import preprocess
from ecg_processing.r_peak_detection import (
    detect_rpeaks, get_rr_intervals, get_heart_rate, get_hrv_metrics
)


def test_rpeaks(record_name='100'):
    # ── Load and preprocess ───────────────────────────────────────
    print(f"Loading record {record_name}...")
    signal, fs, labels = load_record(f'data/raw/mitdb/{record_name}')

    # Use 10 seconds, lead 0 (MLII) — must be 1D
    ten_sec = int(10 * fs)
    raw = signal[:ten_sec, 0]
    clean = preprocess(raw, fs)

    # ── Detect R-peaks ────────────────────────────────────────────
    print("Detecting R-peaks...")
    r_peaks = detect_rpeaks(clean, fs)
    print(f"R-peaks found: {len(r_peaks)}")
    print(f"R-peak indices (first 5): {r_peaks[:5]}")

    # ── Compute RR intervals and heart rate ───────────────────────
    rr_intervals = get_rr_intervals(r_peaks, fs)
    heart_rate   = get_heart_rate(rr_intervals)
    hrv          = get_hrv_metrics(rr_intervals)

    print(f"\nHeart Rate : {heart_rate:.1f} BPM")
    print(f"Mean RR    : {np.mean(rr_intervals):.3f} seconds")
    print(f"HRV SDNN   : {hrv['sdnn']}")
    print(f"HRV RMSSD  : {hrv['rmssd']}")

    # ── Plot ──────────────────────────────────────────────────────
    plt.figure(figsize=(14, 4))
    plt.plot(clean, color='blue', linewidth=0.8, label='Clean ECG')
    plt.scatter(
        r_peaks, clean[r_peaks],
        color='red', s=60, zorder=5, label='R-peaks'
    )
    plt.title(f'R-peak Detection — Record {record_name} | HR: {heart_rate:.1f} BPM')
    plt.xlabel('Samples')
    plt.ylabel('Normalised Amplitude')
    plt.legend()
    plt.tight_layout()

    os.makedirs('data/processed', exist_ok=True)
    plt.savefig('data/processed/rpeak_validation.png', dpi=150)
    plt.show()
    print("\nPlot saved to data/processed/rpeak_validation.png")

    # ── Validation checks ─────────────────────────────────────────
    print("\n--- Validation Checks ---")
    assert len(r_peaks) > 0,                     "FAIL: No R-peaks detected"
    assert 40 < heart_rate < 180,                "FAIL: Heart rate out of normal range"
    assert all(rr > 0.2 for rr in rr_intervals), "FAIL: RR interval too short (< 0.3s)"
    assert all(rr < 2.0 for rr in rr_intervals), "FAIL: RR interval too long (> 2.0s)"
    assert not np.any(np.isnan(rr_intervals)),   "FAIL: NaN in RR intervals"

    print("All checks passed")
    return r_peaks, rr_intervals, heart_rate


if __name__ == '__main__':
    test_rpeaks('100')