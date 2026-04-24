import sys
import os
sys.path.append(os.path.abspath('.'))

from ecg_processing.load_ecg import load_record
from ecg_processing.preprocess import preprocess
from ecg_processing.r_peak_detection import (
    detect_rpeaks, get_rr_intervals, get_heart_rate, get_hrv_metrics
)
from ecg_processing.features import extract_features, extract_segments
from ecg_processing.export_to_json import export_to_json


def run_pipeline(record_name, duration_sec=None, save=True):
    """
    Full pipeline: load → preprocess → R-peaks → features → export
    Input:  record name (e.g. '100') and duration in seconds
    Output: exported JSON dict
    """
    print(f"\n>>> Processing record {record_name}...")

    # 1. Load
    signal, fs, labels = load_record(f'data/raw/mitdb/{record_name}')
    raw = signal[:, 0] if duration_sec is None else signal[:int(duration_sec * fs), 0]

    # 2. Preprocess
    clean = preprocess(raw, fs)

    # 3. R-peaks
    r_peaks      = detect_rpeaks(clean, fs)
    rr_intervals = get_rr_intervals(r_peaks, fs)
    heart_rate   = get_heart_rate(rr_intervals)
    hrv          = get_hrv_metrics(rr_intervals)

    # 4. Features
    features = extract_features(r_peaks, rr_intervals, heart_rate, hrv)

    # 5. Export
    result = export_to_json(
        record_id    = record_name,
        signal       = clean,
        fs           = fs,
        r_peaks      = r_peaks,
        rr_intervals = rr_intervals,
        heart_rate   = heart_rate,
        hrv          = hrv,
        features     = features,
        save         = save
    )

    print(f"    Heart Rate : {features['heart_rate']} BPM")
    print(f"    Beat Count : {features['beat_count']}")
    print(f"    R-peaks    : {len(r_peaks)}")
    print(f"    Status     : OK")

    return result


def run_batch(records, duration_sec=10):
    """
    Run pipeline on multiple records.
    Logs failures without crashing — failed records go in the report.
    """
    results  = {}
    failed   = []

    for record in records:
        try:
            results[record] = run_pipeline(record, duration_sec)
        except Exception as e:
            print(f"    FAILED: {e}")
            failed.append({"record": record, "error": str(e)})

    print(f"\n--- Batch Complete ---")
    print(f"Succeeded : {len(results)}")
    print(f"Failed    : {len(failed)}")

    if failed:
        print("\nFailed records:")
        for f in failed:
            print(f"  {f['record']} — {f['error']}")

    return results, failed


if __name__ == '__main__':
    # Single record test
    if len(sys.argv) > 1:
        run_pipeline(sys.argv[1])
    else:
        # Default: run batch on 10 records
        records = ['100', '101', '102', '103', '104',
                   '105', '106', '107', '108', '109']
        run_batch(records)