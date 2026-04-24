"""
core/loader/csv_parser.py
─────────────────────────────────────────────────────────────────────────────
Universal Preprocessor for Hospital-style ECG CSV files.

Tested against:
  - "A Large Scale 12-lead ECG Database for Arrhythmia Study" (PhysioNet)
    aka the Chapman-Shaoxing / CPSC-Extra dataset.
    Format: 5000 rows × 12 cols, headers = ['I','II','III','aVR','aVL','aVF',
             'V1','V2','V3','V4','V5','V6'], unit = microvolts (µV), 500 Hz.

  - Generic GE MUSE XML → CSV export
    Common formats include a leading TIME column and vendor-prefixed lead names.

Pipeline
────────
  1. Vendor Detection  – inspect header strings for known vendor fingerprints
  2. Time-column Strip – remove any sample-clock / time axis column
  3. Lead Mapping      – normalise aliases to the 12 canonical lead names
  4. Missing Lead Fill – zero-pad any missing lead so the AI never crashes
  5. Unit Conversion   – detect µV and rescale to mV  (threshold: |max| > 50)
  6. Shape Guarantee   – return ndarray (n_samples, 12), int sampling_rate
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

# Canonical lead order expected by FCN Wang and NeuroKit2
REQUIRED_LEADS: list[str] = [
    "I", "II", "III", "AVR", "AVL", "AVF",
    "V1", "V2", "V3", "V4", "V5", "V6",
]

# ── Lead alias map (keys are upper-cased before lookup) ──────────────────────
# Covers: PhysioNet large-scale DB, GE MUSE XML export, Philips, Schiller,
#         numbered ECG columns, and common research variants.
LEAD_ALIASES: dict[str, str] = {
    # Standard short names (already correct, but we still upper-case them)
    "I": "I", "II": "II", "III": "III",
    "AVR": "AVR", "AVL": "AVL", "AVF": "AVF",
    "V1": "V1", "V2": "V2", "V3": "V3",
    "V4": "V4", "V5": "V5", "V6": "V6",

    # aVx lower-case variants (PhysioNet large-scale DB uses these)
    "AVR": "AVR", "AVL": "AVL", "AVF": "AVF",

    # GE MUSE / Philips full-word variants
    "LEAD I":   "I",   "LEAD II":   "II",   "LEAD III": "III",
    "LEAD AVR": "AVR", "LEAD AVL":  "AVL",  "LEAD AVF": "AVF",
    "LEAD V1":  "V1",  "LEAD V2":   "V2",   "LEAD V3":  "V3",
    "LEAD V4":  "V4",  "LEAD V5":   "V5",   "LEAD V6":  "V6",

    # GE MUSE "ECG" prefix
    "ECG I":   "I",   "ECG II":   "II",   "ECG III": "III",
    "ECG AVR": "AVR", "ECG AVL":  "AVL",  "ECG AVF": "AVF",
    "ECG V1":  "V1",  "ECG V2":   "V2",   "ECG V3":  "V3",
    "ECG V4":  "V4",  "ECG V5":   "V5",   "ECG V6":  "V6",

    # ECG1–12 numeric convention
    "ECG1":  "I",   "ECG2":  "II",   "ECG3":  "III",
    "ECG4":  "AVR", "ECG5":  "AVL",  "ECG6":  "AVF",
    "ECG7":  "V1",  "ECG8":  "V2",   "ECG9":  "V3",
    "ECG10": "V4",  "ECG11": "V5",   "ECG12": "V6",

    # Schiller "C" chest-lead convention
    "C1": "V1", "C2": "V2", "C3": "V3",
    "C4": "V4", "C5": "V5", "C6": "V6",

    # Averaged / filtered variants
    "V1-AVG": "V1", "V2-AVG": "V2", "V3-AVG": "V3",
    "V4-AVG": "V4", "V5-AVG": "V5", "V6-AVG": "V6",
    "I-AVG":  "I",  "II-AVG": "II",

    # Verbose augmented names
    "AUG VOLTAGE RIGHT":    "AVR",
    "AUG VOLTAGE LEFT":     "AVL",
    "AUG VOLTAGE FOOT":     "AVF",
    "AUGMENTED RIGHT":      "AVR",
    "AUGMENTED LEFT":       "AVL",
    "AUGMENTED FOOT":       "AVF",
}

# Vendor fingerprint strings that appear in CSV headers (upper-cased search)
VENDOR_FINGERPRINTS: dict[str, list[str]] = {
    "GE_MUSE":   ["GE", "MUSE", "MAC ", "MARQUETTE"],
    "PHILIPS":   ["PHILIPS", "PAGEWRITER", "TC70", "TC30"],
    "SCHILLER":  ["SCHILLER", "CARDIOVIT"],
    "NIHON":     ["NIHON", "KOHDEN", "NK ECG"],
    "PHYSIONET": ["PHYSIONET", "RECORD", "CPSC"],
}

# Column names that are definitely not lead data
TIME_COLUMN_NAMES: set[str] = {
    "TIME", "TIMESTAMP", "T", "SAMPLE", "SAMPLE_INDEX",
    "INDEX", "IDX", "MS", "SEC", "SECONDS", "MILLISECONDS",
    "ELAPSED", "ELAPSED_TIME", "FRAME",
}

# µV → mV threshold: any signal whose absolute max exceeds this is assumed µV
MICROVOLT_THRESHOLD: float = 50.0


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

class CSVParseResult:
    """Container returned by load_ecg_csv — carries the signal and metadata."""

    def __init__(
        self,
        signal: np.ndarray,
        sampling_rate: int,
        vendor: str,
        imputed_leads: list[str],
        unit_converted: bool,
        original_columns: list[str],
    ) -> None:
        self.signal = signal                    # (n_samples, 12)  float32, mV
        self.sampling_rate = sampling_rate      # Hz
        self.vendor = vendor                    # detected vendor string
        self.imputed_leads = imputed_leads      # leads that were zero-padded
        self.unit_converted = unit_converted    # True if µV → mV conversion ran
        self.original_columns = original_columns  # raw headers before mapping

    def to_notes(self) -> list[str]:
        """Human-readable notes suitable for the AnalysisResponse.notes field."""
        notes: list[str] = []
        notes.append(f"CSV source detected as: {self.vendor}")
        if self.unit_converted:
            notes.append(
                "Amplitude values exceeded 50 mV — "
                "converted from microvolts (µV) to millivolts (mV)."
            )
        if self.imputed_leads:
            leads_str = ", ".join(self.imputed_leads)
            notes.append(
                f"Missing leads zero-padded for model compatibility: {leads_str}"
            )
        return notes


def load_ecg_csv(
    file_path: str | Path,
    assumed_sr: int = 500,
) -> tuple[np.ndarray, int]:
    """
    Minimal entry-point that matches the signature expected by ecg_loader.py:

        signal_raw, native_sr = load_ecg_csv(path)

    Returns
    -------
    signal_raw : np.ndarray, shape (n_samples, 12), dtype float32, unit mV
    native_sr  : int  — sampling rate in Hz (assumed_sr unless header says otherwise)
    """
    result = load_ecg_csv_full(file_path, assumed_sr=assumed_sr)
    return result.signal, result.sampling_rate


def load_ecg_csv_full(
    file_path: str | Path,
    assumed_sr: int = 500,
) -> CSVParseResult:
    """
    Full-detail version — returns a CSVParseResult with provenance metadata.
    Use this when you want to forward notes / warnings into AnalysisResponse.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"ECG CSV not found: {path}")

    # ── Step 0: Sniff raw header for vendor + sampling-rate hints ────────────
    raw_header, detected_sr = _sniff_header(path, assumed_sr)
    vendor = _detect_vendor(raw_header)
    logger.info("CSV load: vendor=%s  assumed_sr=%d  file=%s", vendor, detected_sr, path.name)

    # ── Step 1: Read CSV into DataFrame ──────────────────────────────────────
    df = _read_csv_robust(path)
    original_columns = list(df.columns)

    # ── Step 2: Strip non-signal columns (time, index, …) ────────────────────
    df = _strip_non_signal_columns(df)

    # ── Step 3: Normalise column names → canonical lead names ────────────────
    df = _map_lead_columns(df)

    # ── Step 4: Build 12-column matrix, zero-padding missing leads ───────────
    signal_raw, imputed_leads = _assemble_12_leads(df)

    # ── Step 5: Unit conversion µV → mV ──────────────────────────────────────
    unit_converted = False
    if np.max(np.abs(signal_raw)) > MICROVOLT_THRESHOLD:
        signal_raw = signal_raw / 1000.0
        unit_converted = True
        logger.info("Unit conversion applied (µV → mV) for %s", path.name)

    # ── Step 6: Sanity-check final shape ─────────────────────────────────────
    if signal_raw.shape[1] != 12:
        raise ValueError(
            f"Unexpected shape after assembly: {signal_raw.shape}. "
            "Expected (n_samples, 12)."
        )

    signal_raw = signal_raw.astype(np.float32)

    return CSVParseResult(
        signal=signal_raw,
        sampling_rate=detected_sr,
        vendor=vendor,
        imputed_leads=imputed_leads,
        unit_converted=unit_converted,
        original_columns=original_columns,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Private helpers
# ─────────────────────────────────────────────────────────────────────────────

def _sniff_header(path: Path, fallback_sr: int) -> tuple[str, int]:
    """
    Read the first few raw lines looking for:
      - Vendor fingerprint strings
      - Embedded sampling-rate metadata  (e.g. "Sampling Rate: 500")

    Returns (raw_header_text, sampling_rate).
    """
    raw_lines: list[str] = []
    sr = fallback_sr
    try:
        with path.open("r", encoding="utf-8", errors="replace") as fh:
            for i, line in enumerate(fh):
                if i >= 10:
                    break
                raw_lines.append(line)
                # Look for "sampling rate" annotation lines common in GE MUSE exports
                upper = line.upper()
                if "SAMPLING" in upper and "RATE" in upper:
                    # Try to extract the integer value
                    import re
                    match = re.search(r"(\d{3,4})", line)
                    if match:
                        candidate = int(match.group(1))
                        if 100 <= candidate <= 2000:
                            sr = candidate
                            logger.info("Detected embedded sampling rate: %d Hz", sr)
    except Exception as exc:
        logger.warning("Header sniff failed (%s) — using fallback sr=%d", exc, fallback_sr)

    return "".join(raw_lines), sr


def _detect_vendor(header_text: str) -> str:
    """Return a human-readable vendor string based on header fingerprints."""
    upper = header_text.upper()
    for vendor_name, fingerprints in VENDOR_FINGERPRINTS.items():
        if any(fp in upper for fp in fingerprints):
            return vendor_name
    return "UNKNOWN / Generic CSV"


def _read_csv_robust(path: Path) -> pd.DataFrame:
    """
    Read a CSV tolerating:
      - BOM (byte-order mark) from Windows exports
      - Mixed encodings
      - Metadata rows before the actual header (GE MUSE often has 1-3 such rows)
      - Extra trailing whitespace in values
    """
    encodings = ["utf-8-sig", "utf-8", "latin-1", "cp1252"]

    for enc in encodings:
        try:
            df = _try_read_csv(path, enc)
            if df is not None and len(df.columns) >= 2:
                logger.debug("CSV read OK with encoding=%s, shape=%s", enc, df.shape)
                return df
        except Exception:
            continue

    raise ValueError(f"Could not parse CSV file: {path}. Tried encodings: {encodings}")


def _try_read_csv(path: Path, encoding: str) -> Optional[pd.DataFrame]:
    """
    Attempt a single CSV read.  If the first row looks like metadata rather
    than a data header, skip rows until we find the real column row.
    """
    # First try: standard read
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df = pd.read_csv(path, encoding=encoding, skipinitialspace=True)

    # Sanity check: if columns are unnamed integers, the header row is probably
    # buried under metadata lines. Try skipping 1–4 rows.
    if _columns_look_like_data(df.columns):
        for skip in range(1, 5):
            try:
                df2 = pd.read_csv(
                    path, encoding=encoding, skiprows=skip, skipinitialspace=True
                )
                if not _columns_look_like_data(df2.columns) and len(df2.columns) >= 2:
                    logger.info("Skipped %d metadata row(s) in %s", skip, path.name)
                    return df2
            except Exception:
                break

    return df


def _columns_look_like_data(columns: pd.Index) -> bool:
    """
    Return True when column names look like raw data values rather than
    human-readable lead names (e.g. Pandas assigned 0, 1, 2 … or '-0.1234').
    """
    for col in columns:
        col_str = str(col).strip()
        try:
            float(col_str)
            return True   # numeric column name → data ended up in header row
        except ValueError:
            pass
    return False


def _strip_non_signal_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Remove time / index columns before lead mapping."""
    # Upper-case column names for comparison, but keep original for drop list
    to_drop = [
        col for col in df.columns
        if col.strip().upper() in TIME_COLUMN_NAMES
    ]
    if to_drop:
        logger.debug("Stripping non-signal columns: %s", to_drop)
        df = df.drop(columns=to_drop)
    return df


def _map_lead_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rename every column that has a known alias to its canonical lead name.
    Columns that cannot be mapped are kept as-is (they will be ignored later
    if they don't appear in REQUIRED_LEADS).
    """
    rename_map: dict[str, str] = {}
    for col in df.columns:
        upper_col = col.strip().upper()
        if upper_col in LEAD_ALIASES:
            canonical = LEAD_ALIASES[upper_col]
            if canonical != col:
                rename_map[col] = canonical
        # Also handle "aVR" → "AVR" style (PhysioNet large-scale DB)
        # The DB uses lowercase 'a' prefix
        elif upper_col.startswith("AV") and len(upper_col) == 3:
            rename_map[col] = upper_col  # e.g. "aVR" → "AVR"

    if rename_map:
        logger.debug("Lead renames applied: %s", rename_map)
        df = df.rename(columns=rename_map)

    # Upper-case all remaining column names for uniform downstream lookup
    df.columns = [c.strip().upper() for c in df.columns]
    return df


def _assemble_12_leads(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """
    Build a (n_samples, 12) float64 array in canonical lead order.
    Any lead missing from df is filled with zeros and recorded in imputed_leads.
    """
    n_samples = len(df)
    imputed_leads: list[str] = []

    columns: dict[str, np.ndarray] = {}
    for lead in REQUIRED_LEADS:
        if lead in df.columns:
            col_data = pd.to_numeric(df[lead], errors="coerce").fillna(0.0).values
            columns[lead] = col_data
        else:
            logger.warning(
                "Lead %s not found in CSV — zero-padding for model compatibility.", lead
            )
            columns[lead] = np.zeros(n_samples, dtype=np.float64)
            imputed_leads.append(lead)

    signal = np.column_stack([columns[lead] for lead in REQUIRED_LEADS])
    return signal, imputed_leads