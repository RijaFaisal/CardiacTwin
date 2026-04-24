"""
Cross-Dataset Generalisation Evaluation
========================================
Runs the trained CNN (MIT-BIH only) zero-shot on PTB-XL and PTB-DB records
and measures agreement between the model's dominant beat prediction and the
record-level ground-truth diagnosis.

This is Option B from the multi-dataset analysis: rather than training on
incompatible datasets, we use them as held-out sets to test generalisation.

Usage (from repo root, venv activated):
    python ml/cross_dataset_eval.py
    python ml/cross_dataset_eval.py --max-records 50   # quick smoke-test

Output:
    ml/results/cross_dataset/ptbxl_results.csv
    ml/results/cross_dataset/ptbdb_results.csv
    ml/results/cross_dataset/agreement_summary.png
    ml/results/cross_dataset/report.txt
"""

import sys
import os
sys.path.append(os.path.abspath('.'))

import argparse
import ast
from pathlib import Path

import numpy as np
import pandas as pd
import wfdb
from scipy.signal import resample_poly
from tensorflow import keras
import matplotlib.pyplot as plt

from ecg_processing.preprocess import preprocess
from ecg_processing.r_peak_detection import detect_rpeaks
from ecg_processing.features import extract_segments

# ── Constants ─────────────────────────────────────────────────────────────────

TARGET_FS      = 360          # CNN was trained at 360 Hz (MIT-BIH)
SEGMENT_WINDOW = 180          # 180 samples @ 360 Hz ≈ 500 ms per beat
DURATION_SEC   = 10           # Use first 10 seconds of each record
MODEL_PATH     = 'models/arrhythmia_cnn_best.h5'

PTBXL_BASE = Path('data/raw/ptbxl/physionet.org/files/ptb-xl/1.0.3')
PTBDB_BASE = Path('data/raw/ptbdb')
OUT_DIR    = Path('ml/results/cross_dataset')

CLASS_NAMES = {
    0: 'Normal',
    1: 'Bundle Branch Block',
    2: 'Ventricular',
    3: 'Atrial',
    4: 'Other',
}

# ── PTB-XL SCP code → 5-class mapping ────────────────────────────────────────
# Only codes that map cleanly to one of our beat-level classes are included.
# All unlisted codes (MI subtypes, HYP, STTC, conduction intervals, etc.)
# fall through to Other (4) — see get_ptbxl_ground_truth().
PTBXL_SCP_MAP = {
    # ── Normal ──────────────────────────────────────────────────────────────
    'NORM':  0,   # Normal ECG
    'SR':    0,   # Sinus rhythm

    # ── Bundle Branch Block ──────────────────────────────────────────────────
    'CLBBB': 1,   # Complete LBBB
    'CRBBB': 1,   # Complete RBBB
    'ILBBB': 1,   # Incomplete LBBB
    'IRBBB': 1,   # Incomplete RBBB
    'IVCD':  1,   # Intraventricular conduction delay

    # ── Ventricular ─────────────────────────────────────────────────────────
    'PVC':   2,   # Premature ventricular contraction
    'BIGU':  2,   # Bigeminy
    'TRIGU': 2,   # Trigeminy
    'VT':    2,   # Ventricular tachycardia
    'VFIB':  2,   # Ventricular fibrillation

    # ── Atrial ──────────────────────────────────────────────────────────────
    'AFIB':  3,   # Atrial fibrillation
    'AFLT':  3,   # Atrial flutter
    'PAC':   3,   # Premature atrial contraction
    'PSVT':  3,   # Paroxysmal supraventricular tachycardia
    'SVTAC': 3,   # Supraventricular tachycardia
}

# Priority order when a record has multiple mapped codes (highest priority first)
CLASS_PRIORITY = [2, 3, 1, 0, 4]  # Ventricular > Atrial > BBB > Normal > Other

# ── PTB-DB: reason for admission → 5-class mapping ───────────────────────────
# The downloaded PTB-DB subset contains almost exclusively MI and Stable angina.
# Both are structural/ischaemic — they affect ST morphology, not beat shape.
# They correctly map to Other (4); this documents an expected CNN limitation.
PTBDB_DIAGNOSIS_MAP = {
    'healthy control':       0,
    'bundle branch block':   1,
    'dysrhythmia':           4,   # Too broad to sub-classify without beat labels
    'myocardial infarction': 4,
    'stable angina':         4,
    'cardiomyopathy':        4,
    'heart failure':         4,
    'hypertension':          4,
    'valvular heart disease': 4,
    'myocarditis':           4,
}


# ── Model ─────────────────────────────────────────────────────────────────────

def load_model():
    print(f"Loading model from {MODEL_PATH} ...")
    model = keras.models.load_model(MODEL_PATH)
    print("Model loaded.\n")
    return model


# ── Resampling ────────────────────────────────────────────────────────────────

def resample_to_target(signal_1d: np.ndarray, original_fs: float) -> np.ndarray:
    """
    Resample a 1-D ECG signal to TARGET_FS using polyphase filtering.
    Polyphase resampling applies anti-aliasing automatically.

    PTB-XL 100 Hz → 360 Hz : up=18, down=5   (GCD(100,360)=20)
    PTB-DB 1000 Hz → 360 Hz: up=9,  down=25  (GCD(1000,360)=40)
    """
    if original_fs == TARGET_FS:
        return signal_1d

    fs = int(original_fs)
    target = int(TARGET_FS)
    from math import gcd
    g = gcd(fs, target)
    return resample_poly(signal_1d, up=target // g, down=fs // g)


# ── Inference ─────────────────────────────────────────────────────────────────

def run_inference(signal_1d: np.ndarray, original_fs: float, model) -> dict | None:
    """
    Full inference pipeline on a single-lead ECG:
      resample → preprocess → R-peak detection → segment → CNN → dominant class

    Returns a result dict, or None if R-peak detection produces no segments.
    """
    # 1. Clip to DURATION_SEC
    n_samples = int(DURATION_SEC * original_fs)
    raw = signal_1d[:n_samples]

    # 2. Resample to TARGET_FS
    resampled = resample_to_target(raw, original_fs)

    # 3. Preprocess (bandpass → baseline removal → z-score)
    try:
        clean = preprocess(resampled, TARGET_FS)
    except ValueError:
        return None   # flat/corrupt signal

    # 4. R-peak detection
    r_peaks = detect_rpeaks(clean, TARGET_FS)
    if len(r_peaks) == 0:
        return None

    # 5. Beat segmentation
    segments, _ = extract_segments(clean, r_peaks, window=SEGMENT_WINDOW)
    if len(segments) == 0:
        return None

    # 6. CNN inference
    X = segments[..., np.newaxis]               # (N, 180, 1)
    probs = model.predict(X, verbose=0)         # (N, 5)
    pred_classes = np.argmax(probs, axis=1)     # (N,)
    confidences  = np.max(probs, axis=1)        # (N,)

    # 7. Dominant class (most frequent beat class)
    counts = np.bincount(pred_classes, minlength=5)
    dominant_class = int(np.argmax(counts))
    dominant_conf  = float(np.mean(confidences[pred_classes == dominant_class]) * 100)

    beat_dist = {CLASS_NAMES[i]: round(100 * counts[i] / len(pred_classes), 1)
                 for i in range(5)}

    return {
        'dominant_class':      dominant_class,
        'dominant_name':       CLASS_NAMES[dominant_class],
        'dominant_confidence': round(dominant_conf, 2),
        'beats_analysed':      len(segments),
        'beat_distribution':   beat_dist,
    }


# ── PTB-XL ────────────────────────────────────────────────────────────────────

def get_ptbxl_ground_truth(scp_codes: dict) -> tuple[int, str]:
    """
    Map a PTB-XL SCP-codes dict to one of our 5 classes.

    Strategy:
      1. For every code with likelihood > 0, look up PTBXL_SCP_MAP.
      2. Among all matched classes, apply CLASS_PRIORITY ordering.
      3. If no code maps to a known class → Other (4).

    The SCP likelihood scores (0–100) indicate how confident annotators were.
    We use >0 as the threshold so that co-occurring secondary diagnoses are
    considered, but the priority ordering ensures the most clinically
    significant arrhythmia class wins.
    """
    mapped = set()
    for code, score in scp_codes.items():
        if score > 0 and code in PTBXL_SCP_MAP:
            mapped.add(PTBXL_SCP_MAP[code])

    if not mapped:
        return 4, 'Other (unmapped SCP codes)'

    for cls in CLASS_PRIORITY:
        if cls in mapped:
            # Build a readable label from the contributing codes
            contributing = [c for c, s in scp_codes.items()
                            if s > 0 and PTBXL_SCP_MAP.get(c) == cls]
            return cls, '+'.join(contributing)

    return 4, 'Other'


def evaluate_ptbxl(model, max_records: int | None = None) -> pd.DataFrame:
    """
    Evaluate the CNN on all downloaded PTB-XL records.
    """
    print("=" * 60)
    print("PTB-XL Evaluation (100 Hz → 360 Hz resample, Lead II)")
    print("=" * 60)

    # Load metadata CSV
    csv_path = PTBXL_BASE / 'ptbxl_database.csv'
    df_meta  = pd.read_csv(csv_path, index_col='ecg_id')
    df_meta['scp_codes']   = df_meta['scp_codes'].apply(ast.literal_eval)
    df_meta['lr_basename'] = df_meta['filename_lr'].apply(lambda x: Path(x).stem)
    meta_lookup = df_meta.set_index('lr_basename')

    # Discover downloaded records
    records_dir = PTBXL_BASE / 'records100'
    hea_files   = sorted(records_dir.rglob('*.hea'))
    if max_records:
        hea_files = hea_files[:max_records]

    print(f"Records found : {len(hea_files)}")

    rows = []
    for i, hea_path in enumerate(hea_files, 1):
        stem = hea_path.stem

        # Look up ground truth
        if stem not in meta_lookup.index:
            continue
        scp_codes       = meta_lookup.loc[stem, 'scp_codes']
        gt_class, gt_label = get_ptbxl_ground_truth(scp_codes)

        # Load ECG (Lead II = index 1)
        try:
            rec    = wfdb.rdrecord(str(hea_path.with_suffix('')))
            signal = rec.p_signal[:, 1].astype(np.float64)   # Lead II
            fs     = float(rec.fs)
        except Exception as e:
            rows.append(_skip_row(stem, gt_class, gt_label, f'load error: {e}'))
            continue

        # Inference
        result = run_inference(signal, fs, model)
        if result is None:
            rows.append(_skip_row(stem, gt_class, gt_label, 'no segments'))
            continue

        rows.append({
            'record_id':           stem,
            'gt_class':            gt_class,
            'gt_label':            gt_label,
            'gt_name':             CLASS_NAMES[gt_class],
            'pred_class':          result['dominant_class'],
            'pred_name':           result['dominant_name'],
            'agreement':           result['dominant_class'] == gt_class,
            'confidence':          result['dominant_confidence'],
            'beats_analysed':      result['beats_analysed'],
            'beat_dist_normal':    result['beat_distribution']['Normal'],
            'beat_dist_bbb':       result['beat_distribution']['Bundle Branch Block'],
            'beat_dist_ventr':     result['beat_distribution']['Ventricular'],
            'beat_dist_atrial':    result['beat_distribution']['Atrial'],
            'beat_dist_other':     result['beat_distribution']['Other'],
            'skipped':             False,
            'skip_reason':         '',
        })

        if i % 100 == 0:
            done = sum(1 for r in rows if not r['skipped'])
            print(f"  [{i}/{len(hea_files)}] processed {done} records ...")

    df = pd.DataFrame(rows)
    _print_dataset_report(df, 'PTB-XL')
    return df


# ── PTB-DB ────────────────────────────────────────────────────────────────────

def parse_ptbdb_diagnosis(comments: list[str]) -> str | None:
    """Extract 'Reason for admission' from WFDB header comments."""
    for line in comments:
        if 'Reason for admission' in line:
            return line.split(':', 1)[1].strip()
    return None


def get_ptbdb_ground_truth(diagnosis: str | None) -> tuple[int, str]:
    """Map a PTB-DB diagnosis string to one of our 5 classes."""
    if diagnosis is None:
        return 4, 'Unknown'
    key = diagnosis.lower().strip()
    for pattern, cls in PTBDB_DIAGNOSIS_MAP.items():
        if pattern in key:
            return cls, diagnosis
    return 4, f'Other ({diagnosis})'


def evaluate_ptbdb(model, max_records: int | None = None) -> pd.DataFrame:
    """
    Evaluate the CNN on all downloaded PTB-DB records.
    Uses Lead II (index 1), first 10 seconds, resampled from 1000 Hz → 360 Hz.
    """
    print("\n" + "=" * 60)
    print("PTB-DB Evaluation (1000 Hz → 360 Hz resample, Lead II)")
    print("=" * 60)

    hea_files = sorted(PTBDB_BASE.rglob('*.hea'))
    if max_records:
        hea_files = hea_files[:max_records]

    print(f"Records found : {len(hea_files)}")

    rows = []
    for i, hea_path in enumerate(hea_files, 1):
        stem = hea_path.stem

        # Load header first to get diagnosis
        try:
            rec       = wfdb.rdrecord(str(hea_path.with_suffix('')))
            diagnosis = parse_ptbdb_diagnosis(rec.comments)
            gt_class, gt_label = get_ptbdb_ground_truth(diagnosis)
            fs        = float(rec.fs)

            # Lead selection: use Lead II if available (index 1)
            # PTB-DB has up to 15 leads; lead at index 1 is 'ii'
            if rec.p_signal.shape[1] > 1:
                signal = rec.p_signal[:, 1].astype(np.float64)
            else:
                signal = rec.p_signal[:, 0].astype(np.float64)

        except Exception as e:
            rows.append(_skip_row(stem, 4, 'Unknown', f'load error: {e}'))
            continue

        # Inference
        result = run_inference(signal, fs, model)
        if result is None:
            rows.append(_skip_row(stem, gt_class, gt_label, 'no segments'))
            continue

        rows.append({
            'record_id':           stem,
            'gt_class':            gt_class,
            'gt_label':            gt_label,
            'gt_name':             CLASS_NAMES[gt_class],
            'pred_class':          result['dominant_class'],
            'pred_name':           result['dominant_name'],
            'agreement':           result['dominant_class'] == gt_class,
            'confidence':          result['dominant_confidence'],
            'beats_analysed':      result['beats_analysed'],
            'beat_dist_normal':    result['beat_distribution']['Normal'],
            'beat_dist_bbb':       result['beat_distribution']['Bundle Branch Block'],
            'beat_dist_ventr':     result['beat_distribution']['Ventricular'],
            'beat_dist_atrial':    result['beat_distribution']['Atrial'],
            'beat_dist_other':     result['beat_distribution']['Other'],
            'skipped':             False,
            'skip_reason':         '',
        })

        if i % 50 == 0:
            done = sum(1 for r in rows if not r['skipped'])
            print(f"  [{i}/{len(hea_files)}] processed {done} records ...")

    df = pd.DataFrame(rows)
    _print_dataset_report(df, 'PTB-DB')
    return df


# ── Reporting ─────────────────────────────────────────────────────────────────

def _skip_row(record_id, gt_class, gt_label, reason):
    return {
        'record_id':      record_id,
        'gt_class':       gt_class,
        'gt_label':       gt_label,
        'gt_name':        CLASS_NAMES.get(gt_class, 'Unknown'),
        'pred_class':     -1,
        'pred_name':      'N/A',
        'agreement':      False,
        'confidence':     0.0,
        'beats_analysed': 0,
        'beat_dist_normal': 0, 'beat_dist_bbb': 0,
        'beat_dist_ventr': 0,  'beat_dist_atrial': 0,
        'beat_dist_other': 0,
        'skipped':        True,
        'skip_reason':    reason,
    }


def _print_dataset_report(df: pd.DataFrame, name: str):
    valid = df[~df['skipped']]
    skipped = df[df['skipped']]

    print(f"\n--- {name} Summary ---")
    print(f"Total records   : {len(df)}")
    print(f"Evaluated       : {len(valid)}")
    print(f"Skipped         : {len(skipped)}")
    if len(valid) == 0:
        print("No valid records to report.")
        return

    overall = valid['agreement'].mean() * 100
    print(f"Overall agreement: {overall:.1f}%")

    print(f"\nPer ground-truth class:")
    print(f"  {'Class':<25} {'GT':>5} {'Agree':>6} {'Rate':>7}  {'Dominant predicted class'}")
    print(f"  {'-'*70}")

    for cls in range(5):
        subset = valid[valid['gt_class'] == cls]
        if len(subset) == 0:
            continue
        agree = subset['agreement'].sum()
        rate  = 100 * agree / len(subset)
        # Show what the model predicted for this gt class
        pred_dist = subset['pred_name'].value_counts()
        pred_str  = ', '.join(f"{n}:{c}" for n, c in pred_dist.items())
        print(f"  {CLASS_NAMES[cls]:<25} {len(subset):>5} {agree:>6} {rate:>6.1f}%  [{pred_str}]")


def build_report_text(df_ptbxl: pd.DataFrame, df_ptbdb: pd.DataFrame) -> str:
    lines = []
    lines.append("=" * 70)
    lines.append("CROSS-DATASET GENERALISATION REPORT")
    lines.append("Model: CNN trained on MIT-BIH Arrhythmia Database only")
    lines.append("=" * 70)

    for df, name, note in [
        (df_ptbxl, 'PTB-XL', 'Resampled 100 Hz → 360 Hz, Lead II'),
        (df_ptbdb, 'PTB-DB', 'Resampled 1000 Hz → 360 Hz, Lead II'),
    ]:
        valid = df[~df['skipped']]
        lines.append(f"\n{'─'*70}")
        lines.append(f"Dataset : {name}  ({note})")
        lines.append(f"{'─'*70}")
        lines.append(f"Total records   : {len(df)}")
        lines.append(f"Evaluated       : {len(valid)}")
        lines.append(f"Skipped         : {len(df) - len(valid)}")
        if len(valid) == 0:
            lines.append("(no valid records)")
            continue

        overall = valid['agreement'].mean() * 100
        lines.append(f"Overall agreement: {overall:.1f}%\n")
        lines.append(f"  {'Class':<25} {'GT':>5} {'Agree':>6} {'Rate':>7}")
        lines.append(f"  {'-'*45}")
        for cls in range(5):
            subset = valid[valid['gt_class'] == cls]
            if len(subset) == 0:
                continue
            agree = subset['agreement'].sum()
            rate  = 100 * agree / len(subset)
            lines.append(f"  {CLASS_NAMES[cls]:<25} {len(subset):>5} {agree:>6} {rate:>6.1f}%")

        lines.append(f"\nNote: 'Agreement' = CNN dominant beat class matches")
        lines.append(f"       record-level ground-truth diagnosis.")
        lines.append(f"       Lower agreement on structural diseases (MI, HYP) is")
        lines.append(f"       expected — these do not alter individual beat morphology.")

    lines.append(f"\n{'=' * 70}")
    return '\n'.join(lines)


def plot_results(df_ptbxl: pd.DataFrame, df_ptbdb: pd.DataFrame):
    """
    Two-panel figure:
      Left:  PTB-XL per-class agreement bar chart
      Right: PTB-DB per-class agreement bar chart
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        'Cross-Dataset Generalisation\n'
        'CNN trained on MIT-BIH only — evaluated zero-shot on PTB-XL and PTB-DB',
        fontsize=11
    )

    colours = ['#2ecc71', '#3498db', '#e74c3c', '#f39c12', '#95a5a6']

    for ax, df, name in [(axes[0], df_ptbxl, 'PTB-XL'), (axes[1], df_ptbdb, 'PTB-DB')]:
        valid = df[~df['skipped']]
        if len(valid) == 0:
            ax.set_title(f'{name}\n(no valid records)')
            continue

        labels, rates, counts = [], [], []
        for cls in range(5):
            subset = valid[valid['gt_class'] == cls]
            if len(subset) == 0:
                continue
            labels.append(f"{CLASS_NAMES[cls]}\n(n={len(subset)})")
            rates.append(100 * subset['agreement'].mean())
            counts.append(len(subset))

        bars = ax.bar(labels, rates, color=colours[:len(labels)], edgecolor='white', linewidth=0.8)

        # Annotate bars
        for bar, rate in zip(bars, rates):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 1.5,
                    f'{rate:.1f}%',
                    ha='center', va='bottom', fontsize=9, fontweight='bold')

        overall = valid['agreement'].mean() * 100
        ax.axhline(overall, color='black', linestyle='--', linewidth=1.2, alpha=0.7,
                   label=f'Overall: {overall:.1f}%')
        ax.set_ylim(0, 110)
        ax.set_ylabel('Agreement Rate (%)')
        ax.set_title(f'{name} (n={len(valid)} evaluated)')
        ax.legend(fontsize=9)
        ax.tick_params(axis='x', labelsize=8)
        ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    out_path = OUT_DIR / 'agreement_summary.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nPlot saved to {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Cross-dataset generalisation evaluation for the Cardiac Digital Twin CNN.'
    )
    parser.add_argument(
        '--max-records', type=int, default=None, metavar='N',
        help='Limit evaluation to first N records per dataset (useful for smoke-testing).'
    )
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    model = load_model()

    df_ptbxl = evaluate_ptbxl(model, max_records=args.max_records)
    df_ptbdb  = evaluate_ptbdb(model,  max_records=args.max_records)

    # Save CSVs
    ptbxl_csv = OUT_DIR / 'ptbxl_results.csv'
    ptbdb_csv  = OUT_DIR / 'ptbdb_results.csv'
    df_ptbxl.to_csv(ptbxl_csv, index=False)
    df_ptbdb.to_csv(ptbdb_csv, index=False)
    print(f"\nCSVs saved:")
    print(f"  {ptbxl_csv}")
    print(f"  {ptbdb_csv}")

    # Text report
    report = build_report_text(df_ptbxl, df_ptbdb)
    print('\n' + report)

    report_path = OUT_DIR / 'report.txt'
    report_path.write_text(report)
    print(f"\nReport saved to {report_path}")

    # Plot
    plot_results(df_ptbxl, df_ptbdb)


if __name__ == '__main__':
    main()