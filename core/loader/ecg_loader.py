"""
core/loader/ecg_loader.py
─────────────────────────────────────────────────────────────────────────────
Format-Agnostic ECG Loader  ("The Traffic Cop")

This module is the single entry-point for all data loading in the Dual-Pipeline.
analyze.py calls load_session_ecg() and never needs to know which format was
uploaded — it always receives a clean (signal, sampling_rate) tuple.

Supported formats (detection order):
  1. CSV  — GE MUSE, PhysioNet large-scale DB, Philips, Schiller exports
  2. WFDB — PhysioNet .dat / .hea pairs (existing behaviour, unchanged)

Extension point:
  To add EDF, DICOM, or any other format, add a parser module and one
  elif block in _detect_and_load().  Zero changes needed in analyze.py.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import HTTPException

from core.loader.csv_parser import load_ecg_csv_full, CSVParseResult

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def load_session_ecg(
    session_dir: str | Path,
) -> tuple[np.ndarray, int]:
    """
    Minimal entry-point for analyze.py.  Matches the old load_native() signature:

        signal_raw, native_sr = load_session_ecg(session_dir)

    Returns
    -------
    signal_raw : np.ndarray  shape (n_samples, 12), float32, millivolts
    native_sr  : int         sampling rate in Hz
    """
    signal, sr, _ = load_session_ecg_with_notes(session_dir)
    return signal, sr


def load_session_ecg_with_notes(
    session_dir: str | Path,
) -> tuple[np.ndarray, int, list[str]]:
    """
    Extended entry-point that also returns preprocessing notes.
    Pass these notes into AnalysisResponse.notes so clinicians can see
    what transformations were applied to their data.

    Returns
    -------
    signal_raw : np.ndarray  shape (n_samples, 12), float32, millivolts
    native_sr  : int
    notes      : list[str]   human-readable preprocessing log
    """
    session_path = Path(session_dir)
    if not session_path.is_dir():
        raise HTTPException(
            status_code=404,
            detail=f"Session directory not found: {session_dir}",
        )

    files = os.listdir(session_path)
    signal, sr, notes = _detect_and_load(session_path, files)
    return signal, sr, notes


# ─────────────────────────────────────────────────────────────────────────────
# Internal routing logic
# ─────────────────────────────────────────────────────────────────────────────

def _detect_and_load(
    session_path: Path,
    files: list[str],
) -> tuple[np.ndarray, int, list[str]]:
    """
    Inspect the session directory and dispatch to the correct parser.
    Priority: CSV > WFDB (.hea/.dat) > error.
    """

    # ── 1. CSV ────────────────────────────────────────────────────────────────
    csv_file = _find_file(files, ".csv")
    if csv_file:
        csv_path = session_path / csv_file
        logger.info("ECG loader: CSV detected → %s", csv_file)
        try:
            result: CSVParseResult = load_ecg_csv_full(csv_path)
        except Exception as exc:
            raise HTTPException(
                status_code=422,
                detail=f"Failed to parse CSV '{csv_file}': {exc}",
            )
        notes = result.to_notes()
        notes.insert(0, f"Loaded from CSV: {csv_file}")
        return result.signal, result.sampling_rate, notes

    # ── 2. WFDB (.hea + .dat pair) ────────────────────────────────────────────
    hea_file = _find_file(files, ".hea")
    if hea_file:
        base_path = str(session_path / hea_file[:-4])  # strip .hea
        logger.info("ECG loader: WFDB detected → %s", hea_file)
        try:
            signal, sr = _load_wfdb(base_path)
        except Exception as exc:
            raise HTTPException(
                status_code=422,
                detail=f"Failed to parse WFDB record '{hea_file}': {exc}",
            )
        notes = [f"Loaded from WFDB record: {hea_file[:-4]}"]
        return signal, sr, notes

    # ── 3. Nothing recognised ─────────────────────────────────────────────────
    found = ", ".join(files) if files else "(empty directory)"
    raise HTTPException(
        status_code=404,
        detail=(
            "No valid ECG file found in session directory. "
            "Expected a .csv file (GE MUSE / PhysioNet) or "
            "a .hea/.dat WFDB pair. "
            f"Files present: {found}"
        ),
    )


def _find_file(files: list[str], extension: str) -> Optional[str]:
    """Return the first filename with the given extension, case-insensitive."""
    ext_lower = extension.lower()
    return next((f for f in files if f.lower().endswith(ext_lower)), None)


def _load_wfdb(base_path: str) -> tuple[np.ndarray, int]:
    """
    Thin wrapper around the existing wfdb_parser so the import lives in one
    place and can be mocked in tests.
    """
    # This import points at YOUR existing WFDB loader — no changes needed there.
    from core.loader.wfdb_parser import load_native  # type: ignore[import]
    return load_native(base_path)