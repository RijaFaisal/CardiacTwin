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
    Returns per-record data so save_dataset() can split at the patient level.
    duration_sec: how many seconds per record to use (60 = 1 minute)
    """
    record_data = []   # list of {'name', 'segments', 'labels'}
    failed      = []

    for record_name in records:
        record_path = f'data/raw/mitdb/{record_name}'
        print(f"Processing {record_name}...", end=" ")

        try:
            signal, fs, _ = load_record(record_path)
            n_samples = int(duration_sec * fs)
            raw   = signal[:n_samples, 0]
            clean = preprocess(raw, fs)

            r_peaks = detect_rpeaks(clean, fs)
            if len(r_peaks) == 0:
                print("SKIP (no peaks)")
                failed.append(record_name)
                continue

            segments, valid_peaks = extract_segments(clean, r_peaks)
            if len(segments) == 0:
                print("SKIP (no segments)")
                failed.append(record_name)
                continue

            labels = get_beat_labels(record_path, valid_peaks, fs)
            if labels is None:
                print("SKIP (no annotations)")
                failed.append(record_name)
                continue

            min_len  = min(len(segments), len(labels))
            segments = segments[:min_len]
            labels   = labels[:min_len]

            record_data.append({
                'name':     record_name,
                'segments': segments,
                'labels':   labels,
            })
            print(f"OK — {len(segments)} beats")

        except Exception as e:
            print(f"FAILED — {e}")
            failed.append(record_name)

    # Summary over all records combined
    all_y = np.concatenate([r['labels'] for r in record_data])
    print(f"\n--- Dataset Summary ---")
    print(f"Records loaded : {len(record_data)}")
    print(f"Total beats    : {len(all_y)}")
    print(f"\nClass distribution:")
    for cls, name in CLASS_NAMES.items():
        count = np.sum(all_y == cls)
        pct   = 100 * count / len(all_y)
        print(f"  {name:25s}: {count:5d} ({pct:.1f}%)")
    print(f"\nFailed records : {failed}")

    return record_data


def save_dataset(record_data: list):
    """
    Split at the RECORD level (inter-patient split) to prevent data leakage.

    Beats from the same patient cannot appear in both train and test.
    Random beat-level shuffling inflates accuracy by letting the model
    memorise patient-specific signal characteristics.

    70% of records → train | 15% → val | 15% → test
    Beats are shuffled within each split after record assignment.
    """
    os.makedirs('ml/data', exist_ok=True)

    n       = len(record_data)
    rng     = np.random.default_rng(42)
    indices = rng.permutation(n)

    n_train = int(0.70 * n)
    n_val   = int(0.15 * n)

    train_idx = indices[:n_train]
    val_idx   = indices[n_train:n_train + n_val]
    test_idx  = indices[n_train + n_val:]

    def _concat_and_shuffle(idx_list):
        segs   = np.concatenate([record_data[i]['segments'] for i in idx_list])
        labels = np.concatenate([record_data[i]['labels']   for i in idx_list])
        perm   = np.random.permutation(len(segs))
        return segs[perm], labels[perm]

    X_train, y_train = _concat_and_shuffle(train_idx)
    X_val,   y_val   = _concat_and_shuffle(val_idx)
    X_test,  y_test  = _concat_and_shuffle(test_idx)

    np.save('ml/data/X_train.npy', X_train)
    np.save('ml/data/y_train.npy', y_train)
    np.save('ml/data/X_val.npy',   X_val)
    np.save('ml/data/y_val.npy',   y_val)
    np.save('ml/data/X_test.npy',  X_test)
    np.save('ml/data/y_test.npy',  y_test)

    train_records = [record_data[i]['name'] for i in train_idx]
    val_records   = [record_data[i]['name'] for i in val_idx]
    test_records  = [record_data[i]['name'] for i in test_idx]

    print(f"\n--- Inter-patient Split ---")
    print(f"Train records ({len(train_idx)}): {train_records}")
    print(f"Val   records ({len(val_idx)}):   {val_records}")
    print(f"Test  records ({len(test_idx)}):  {test_records}")
    print(f"\nTrain beats : {X_train.shape}")
    print(f"Val   beats : {X_val.shape}")
    print(f"Test  beats : {X_test.shape}")
    print(f"Files saved to ml/data/")


if __name__ == '__main__':
    record_data = build_dataset()
    save_dataset(record_data)