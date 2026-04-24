import sys
import os
sys.path.append(os.path.abspath('.'))

import numpy as np
import matplotlib.pyplot as plt
from ecg_processing.load_ecg import load_record
from ecg_processing.preprocess import preprocess, bandpass_filter, remove_baseline, normalise


def test_preprocess(record_name='100'):
    # ── Load ──────────────────────────────────────────────────────
    print(f"Loading record {record_name}...")
    signal, fs, labels = load_record(f'data/raw/mitdb/{record_name}')
    print(f"fs: {fs} Hz | signal shape: {signal.shape}")

    # One lead only — preprocess.* expects 1D (filtfilt / medfilt per time series)
    ten_sec = int(10 * fs)
    raw = signal[:ten_sec, 0]

    # ── Run each step individually ─────────────────────────────────
    filtered   = bandpass_filter(raw, fs)
    baselined  = remove_baseline(filtered, fs)
    normalised = normalise(baselined)
    clean      = preprocess(raw, fs)   # master function

    # ── Plot ───────────────────────────────────────────────────────
    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)

    axes[0].plot(raw,        color='grey',   linewidth=0.8)
    axes[0].set_title('1. Raw Signal')
    axes[0].set_ylabel('mV')

    axes[1].plot(filtered,   color='blue',   linewidth=0.8)
    axes[1].set_title('2. After Bandpass Filter (0.5–40 Hz)')
    axes[1].set_ylabel('mV')

    axes[2].plot(baselined,  color='orange', linewidth=0.8)
    axes[2].set_title('3. After Baseline Removal')
    axes[2].set_ylabel('mV')

    axes[3].plot(normalised, color='green',  linewidth=0.8)
    axes[3].set_title('4. After Normalisation (z-score)')
    axes[3].set_ylabel('Normalised')

    plt.xlabel('Samples')
    plt.tight_layout()

    os.makedirs('data/processed', exist_ok=True)
    plt.savefig('data/processed/preprocess_validation.png', dpi=150)
    plt.show()
    print("Plot saved to data/processed/preprocess_validation.png")

    # ── Numeric checks ─────────────────────────────────────────────
    print("\n--- Validation Checks ---")
    print(f"Raw     | mean: {np.mean(raw):.4f}  std: {np.std(raw):.4f}")
    print(f"Clean   | mean: {np.mean(clean):.4f}  std: {np.std(clean):.4f}")

    assert len(clean) == len(raw),          "FAIL: output length changed"
    assert not np.any(np.isnan(clean)),     "FAIL: NaN values found in output"
    assert not np.any(np.isinf(clean)),     "FAIL: Inf values found in output"
    assert abs(np.mean(clean)) < 0.05,      "FAIL: mean not close to 0 after normalisation"
    assert abs(np.std(clean) - 1.0) < 0.1, "FAIL: std not close to 1 after normalisation"

    print("All checks passed")
    return clean


if __name__ == '__main__':
    test_preprocess('100')