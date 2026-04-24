import wfdb
import numpy as np
from scipy.signal import resample

def parse_wfdb(base_file_path: str, target_length: int = 1000) -> np.ndarray:
    """
    Reads a WFDB record (.dat/.hea) and forces it into a (target_length, 12) array.
    """
    try:
        # Step 1: The Extraction
        # wfdb.rdrecord expects the file path WITHOUT the extension.
        # It reads the .hea blueprint to unlock the .dat binary.
        record = wfdb.rdrecord(base_file_path)
        data = record.p_signal
        
        # Step 2: The Resampling
        # If the recording isn't exactly 1000 frames (e.g., recorded at 500Hz), 
        # we mathematically shrink or stretch it.
        if data.shape[0] != target_length:
            data = resample(data, target_length)
            
        return data

    except Exception as e:
        raise ValueError(f"Failed to parse WFDB files: {str(e)}")

def load_native(base_file_path: str):
    record = wfdb.rdrecord(base_file_path)
    raw_data = record.p_signal
    native_sr = record.fs
    lead_names = [name.upper() for name in record.sig_name]

    # 1. Enforce 10 seconds exactly
    target_len = native_sr * 10
    if raw_data.shape[0] > target_len:
        raw_data = raw_data[:target_len, :]  # Crop
    elif raw_data.shape[0] < target_len:
        padding = np.zeros((target_len - raw_data.shape[0], raw_data.shape[1]))
        raw_data = np.vstack((raw_data, padding)) # Pad with zeros

    # 2. Map to exactly 12 leads
    REQUIRED_LEADS = ['I', 'II', 'III', 'AVR', 'AVL', 'AVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    standardized_data = np.zeros((target_len, 12))
    
    for i, required_lead in enumerate(REQUIRED_LEADS):
        search_name = required_lead
        if required_lead == 'II' and 'MLII' in lead_names:
            search_name = 'MLII'

        if search_name in lead_names:
            original_index = lead_names.index(search_name)
            standardized_data[:, i] = raw_data[:, original_index]

    # 2.5 Ensure Lead II (Index 1) has data for NeuroKit2
    # If Lead II was missing, we forcefully copy the first available channel into Lead II.
    # This ensures the clinical rhythm measurements don't flatline.
    if not np.any(standardized_data[:, 1]) and raw_data.shape[1] > 0:
        standardized_data[:, 1] = raw_data[:, 0]

    # 3. Fail-safe amplitude check
    if np.max(np.abs(standardized_data)) > 50:
        standardized_data = standardized_data / 1000.0

    return standardized_data, native_sr
