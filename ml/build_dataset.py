import sys
import os
sys.path.append(os.path.abspath('.'))

import numpy as np
import wfdb
from ecg_processing.load_ecg import load_record
from ecg_processing.preprocess import preprocess
from ecg_processing.r_peak_detection import detect_rpeaks
from ecg_processing.features import extract_segments

# ── Label mapping ─────────────────────────────────────────────────
# MIT-BIH annotation symbols → your 5 classes
LABEL_MAP = {
    'N': 0,  # Normal
    'L': 1,  # Left Bundle Branch Block
    'R': 1,  # Right Bundle Branch Block
    'V': 2,  # Ventricular
    'A': 3,  # Atrial
    'f': 4,  # Other / fusion
    'F': 4,
    '!': 4,
    'E': 4,
}

CLASS_NAMES = {
    0: "Normal",
    1: "Bundle Branch Block",
    2: "Ventricular",
    3: "Atrial",
    4: "Other"
}

# All 48 MIT-BIH records
ALL_RECORDS = [
    '100','101','102','103','104','105','106','107',
    '108','109','111','112','113','114','115','116',
    '117','118','119','121','122','123','124','200',
    '201','202','203','205','207','208','209','210',
    '212','213','214','215','217','219','220','221',
    '222','223','228','230','231','232','233','234'
]


def get_beat_labels(record_path, r_peaks, fs, tolerance_ms=75):
    """
    Match detected R-peaks to annotation beat labels.
    tolerance_ms: how close a detected peak must be to an annotation
    to count as a match.
    """
    try:
        ann = wfdb.rdann(record_path, 'atr')
    except Exception:
        return None

    ann_samples = np.array(ann.sample)
    ann_symbols = np.array(ann.symbol)
    tolerance   = int((tolerance_ms / 1000) * fs)
    labels      = []

    for peak in r_peaks:
        # Find closest annotation to this peak
        diffs = np.abs(ann_samples - peak)
        closest_idx = np.argmin(diffs)

        if diffs[closest_idx] <= tolerance:
            symbol = ann_symbols[closest_idx]
            label  = LABEL_MAP.get(symbol, 4)  # unknown → Other
        else:
            label = 4  # no match → Other

        labels.append(label)

    return np.array(labels)


def build_dataset(records=ALL_RECORDS, duration_sec=60):
    """
    Loop through all records, extract segments and labels.
    duration_sec: how many seconds per record to use (60 = 1 minute)
    """
    all_segments = []
    all_labels   = []
    failed       = []

    for record_name in records:
        record_path = f'data/raw/mitdb/{record_name}'
        print(f"Processing {record_name}...", end=" ")

        try:
            # Load
            signal, fs, _ = load_record(record_path)
            n_samples = int(duration_sec * fs)
            raw   = signal[:n_samples, 0]
            clean = preprocess(raw, fs)

            # R-peaks
            r_peaks = detect_rpeaks(clean, fs)
            if len(r_peaks) == 0:
                print("SKIP (no peaks)")
                failed.append(record_name)
                continue

            # Segments
            segments, valid_peaks = extract_segments(clean, r_peaks)
            if len(segments) == 0:
                print("SKIP (no segments)")
                failed.append(record_name)
                continue

            # Labels
            labels = get_beat_labels(record_path, valid_peaks, fs)
            if labels is None:
                print("SKIP (no annotations)")
                failed.append(record_name)
                continue

            # Trim to matching length
            min_len = min(len(segments), len(labels))
            segments = segments[:min_len]
            labels   = labels[:min_len]

            all_segments.append(segments)
            all_labels.append(labels)
            print(f"OK — {len(segments)} beats")

        except Exception as e:
            print(f"FAILED — {e}")
            failed.append(record_name)

    # Combine all records
    X = np.concatenate(all_segments, axis=0)
    y = np.concatenate(all_labels,   axis=0)

    print(f"\n--- Dataset Summary ---")
    print(f"Total segments : {X.shape[0]}")
    print(f"Segment shape  : {X.shape[1]}")
    print(f"Labels shape   : {y.shape}")
    print(f"\nClass distribution:")
    for cls, name in CLASS_NAMES.items():
        count = np.sum(y == cls)
        pct   = 100 * count / len(y)
        print(f"  {name:25s}: {count:5d} ({pct:.1f}%)")

    print(f"\nFailed records : {failed}")
    return X, y


def save_dataset(X, y):
    """
    Save as numpy arrays — fast to load during training.
    70% train / 15% val / 15% test split.
    """
    os.makedirs('ml/data', exist_ok=True)

    # Shuffle
    idx = np.random.permutation(len(X))
    X, y = X[idx], y[idx]

    # Split
    n      = len(X)
    n_train = int(0.70 * n)
    n_val   = int(0.15 * n)

    X_train, y_train = X[:n_train],           y[:n_train]
    X_val,   y_val   = X[n_train:n_train+n_val], y[n_train:n_train+n_val]
    X_test,  y_test  = X[n_train+n_val:],     y[n_train+n_val:]

    np.save('ml/data/X_train.npy', X_train)
    np.save('ml/data/y_train.npy', y_train)
    np.save('ml/data/X_val.npy',   X_val)
    np.save('ml/data/y_val.npy',   y_val)
    np.save('ml/data/X_test.npy',  X_test)
    np.save('ml/data/y_test.npy',  y_test)

    print(f"\n--- Saved ---")
    print(f"Train : {X_train.shape}")
    print(f"Val   : {X_val.shape}")
    print(f"Test  : {X_test.shape}")
    print(f"Files saved to ml/data/")


if __name__ == '__main__':
    X, y = build_dataset()
    save_dataset(X, y)