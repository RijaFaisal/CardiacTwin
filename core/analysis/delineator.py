"""
core/analysis/delineator.py

Wraps NeuroKit2 to extract all physical ECG measurements needed by the
dashboard:

  - P, Q, R, S, T peak indices and amplitudes (per lead)
  - Heart rate (BPM)
  - PR interval (ms)
  - QRS duration (ms)
  - QTc interval (ms, Bazett correction)
  - Signal quality score (0–100 %)

All functions accept a raw 2-D NumPy array of shape (n_leads, n_samples)
and a sampling rate in Hz, and return plain Python dicts / lists so that
Pydantic can serialise them without extra steps.

Lead indexing (0-based) follows the standard 12-lead order:
  0=I, 1=II, 2=III, 3=aVR, 4=aVL, 5=aVF,
  6=V1, 7=V2, 8=V3, 9=V4, 10=V5, 11=V6
"""

from __future__ import annotations

import logging
from typing import Optional

import neurokit2 as nk
import numpy as np

logger = logging.getLogger(__name__)

# Lead II (index 1) is the clinical standard for rhythm analysis.
RHYTHM_LEAD: int = 1

# Normal reference ranges (all in milliseconds unless noted)
NORMAL_RANGES = {
    "heart_rate_bpm":   (60.0,  100.0),
    "pr_interval_ms":   (120.0, 200.0),
    "qrs_duration_ms":  (70.0,  110.0),
    "qtc_ms":           (350.0, 440.0),
    "signal_quality_pct": (70.0, 100.0),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _status(value: Optional[float], key: str) -> str:
    if value is None:
        return "unknown"
    lo, hi = NORMAL_RANGES[key]
    if value < lo:
        return "low"
    if value > hi:
        return "high"
    return "normal"


# ---------------------------------------------------------------------------
# Peak delineation
# ---------------------------------------------------------------------------

def delineate_peaks(
    signal_2d: np.ndarray,
    sampling_rate: int,
    lead_index: int = RHYTHM_LEAD,
) -> dict:
    """
    Run NeuroKit2 ECG delineation on a single lead.

    signal_2d shape: (n_leads, n_samples)
    sampling_rate: must be the TRUE sampling rate of the signal passed in.
                   Do NOT pass a downsampled signal with the original rate.
    """
    lead = signal_2d[lead_index].astype(float)

    logger.info(
        "Delineating lead %d: %d samples at %d Hz (%.1f s)",
        lead_index, len(lead), sampling_rate, len(lead) / sampling_rate,
    )

    try:
        # Clean the signal first — improves peak detection reliability
        lead_clean = nk.ecg_clean(lead, sampling_rate=sampling_rate)

        _, rpeaks_info = nk.ecg_peaks(lead_clean, sampling_rate=sampling_rate)
        r_peaks = rpeaks_info["ECG_R_Peaks"].tolist()

        logger.info("R-peaks found: %d", len(r_peaks))

        if len(r_peaks) < 2:
            logger.warning(
                "Too few R-peaks (%d). Signal may be too short or noisy.", len(r_peaks)
            )
            return _empty_peaks(r_peaks)

        _, waves_info = nk.ecg_delineate(
            lead_clean,
            rpeaks_info,
            sampling_rate=sampling_rate,
            method="dwt",
            show=False,
        )

    except Exception as exc:
        logger.error("ecg_delineate failed: %s", exc)
        return _empty_peaks([])

    def _clean(key: str) -> list[int]:
        raw = waves_info.get(key, [])
        try:
            arr = np.array(raw, dtype=float)
            return [int(x) for x in arr if not np.isnan(x)]
        except Exception:
            return []

    p_peaks   = _clean("ECG_P_Peaks")
    q_peaks   = _clean("ECG_Q_Peaks")
    s_peaks   = _clean("ECG_S_Peaks")
    t_peaks   = _clean("ECG_T_Peaks")
    p_onsets  = _clean("ECG_P_Onsets")
    p_offsets = _clean("ECG_P_Offsets")
    t_offsets = _clean("ECG_T_Offsets")

    def _mean_amp(indices: list[int]) -> Optional[float]:
        if not indices:
            return None
        amps = [float(lead[i]) for i in indices if 0 <= i < len(lead)]
        return round(float(np.mean(amps)), 4) if amps else None

    return {
        "r_peaks":   r_peaks,
        "p_peaks":   p_peaks,
        "q_peaks":   q_peaks,
        "s_peaks":   s_peaks,
        "t_peaks":   t_peaks,
        "p_onsets":  p_onsets,
        "p_offsets": p_offsets,
        "t_offsets": t_offsets,
        "amplitudes": {
            "p": _mean_amp(p_peaks),
            "q": _mean_amp(q_peaks),
            "r": _mean_amp(r_peaks),
            "s": _mean_amp(s_peaks),
            "t": _mean_amp(t_peaks),
        },
    }


def _empty_peaks(r_peaks: list) -> dict:
    return {
        "r_peaks": r_peaks,
        "p_peaks": [], "q_peaks": [], "s_peaks": [], "t_peaks": [],
        "p_onsets": [], "p_offsets": [], "t_offsets": [],
        "amplitudes": {"p": None, "q": None, "r": None, "s": None, "t": None},
    }


# ---------------------------------------------------------------------------
# Interval calculations
# ---------------------------------------------------------------------------

def calculate_intervals(peaks: dict, sampling_rate: int) -> dict:
    """
    Compute PR interval, QRS duration, QT interval, QTc (Bazett), and
    heart rate from the peak index arrays produced by delineate_peaks().

    All returned intervals are in milliseconds.
    Bazett formula:  QTc = QT / sqrt(RR_seconds)
    """
    ms_per_sample = 1000.0 / sampling_rate

    r_peaks   = np.array(peaks["r_peaks"],   dtype=float)
    p_onsets  = np.array(peaks["p_onsets"],  dtype=float)
    q_peaks   = np.array(peaks["q_peaks"],   dtype=float)
    s_peaks   = np.array(peaks["s_peaks"],   dtype=float)
    t_offsets = np.array(peaks["t_offsets"], dtype=float)

    # ── Heart rate ────────────────────────────────────────────────────────
    heart_rate: Optional[float] = None
    if len(r_peaks) >= 2:
        rr_samples = np.diff(r_peaks)
        rr_ms      = rr_samples * ms_per_sample
        heart_rate = round(60_000.0 / float(np.mean(rr_ms)), 1)

    # ── PR interval  (P onset → R peak) ──────────────────────────────────
    pr_ms: Optional[float] = None
    if len(p_onsets) >= 1 and len(r_peaks) >= 1:
        pr_samples = []
        for r in r_peaks:
            before = p_onsets[p_onsets < r]
            if len(before):
                pr_samples.append(r - before[-1])
        if pr_samples:
            pr_ms = round(float(np.mean(pr_samples)) * ms_per_sample, 1)

    # ── QRS duration  (Q peak → S peak) ──────────────────────────────────
    qrs_ms: Optional[float] = None
    if len(q_peaks) >= 1 and len(s_peaks) >= 1:
        qrs_samples = []
        for q in q_peaks:
            after = s_peaks[s_peaks > q]
            if len(after):
                qrs_samples.append(after[0] - q)
        if qrs_samples:
            qrs_ms = round(float(np.mean(qrs_samples)) * ms_per_sample, 1)

    # ── QT interval  (Q peak → T offset) ─────────────────────────────────
    qt_ms:  Optional[float] = None
    qtc_ms: Optional[float] = None
    if len(q_peaks) >= 1 and len(t_offsets) >= 1 and len(r_peaks) >= 2:
        qt_samples = []
        for q in q_peaks:
            after = t_offsets[t_offsets > q]
            if len(after):
                qt_samples.append(after[0] - q)
        if qt_samples:
            qt_ms = round(float(np.mean(qt_samples)) * ms_per_sample, 1)
            rr_mean_s = float(np.mean(np.diff(r_peaks))) / sampling_rate
            if rr_mean_s > 0:
                qtc_ms = round(qt_ms / np.sqrt(rr_mean_s), 1)

    return {
        "heart_rate_bpm":  heart_rate,
        "pr_interval_ms":  pr_ms,
        "qrs_duration_ms": qrs_ms,
        "qt_interval_ms":  qt_ms,
        "qtc_ms":          qtc_ms,
    }


# ---------------------------------------------------------------------------
# Signal quality
# ---------------------------------------------------------------------------

def calculate_quality(
    signal_2d: np.ndarray,
    sampling_rate: int,
    lead_index: int = RHYTHM_LEAD,
) -> dict:
    """
    Run NeuroKit2 ecg_quality on the rhythm lead.

    Uses 'zhao2018' method which is more reliable across diverse signals
    than 'averageQRS' (which requires pre-detected peaks).
    Returns score_pct 0–100.
    """
    lead = signal_2d[lead_index].astype(float)

    try:
        # zhao2018 returns a single string: 'Excellent', 'Barely', or 'Unacceptable'
        quality_label = nk.ecg_quality(
            lead,
            sampling_rate=sampling_rate,
            method="zhao2018",
        )
        score_map = {
            "Excellent":    100.0,
            "Barely":        50.0,
            "Unacceptable":   0.0,
        }
        score_pct = score_map.get(str(quality_label), 50.0)

    except Exception:
        # Fall back to averageQRS method if zhao2018 fails
        try:
            quality_arr = nk.ecg_quality(
                lead,
                sampling_rate=sampling_rate,
                method="averageQRS",
            )
            score_pct = round(float(np.mean(quality_arr)) * 100, 1)
        except Exception as exc:
            logger.warning("Both ecg_quality methods failed: %s", exc)
            score_pct = 0.0

    return {
        "score_pct": score_pct,
        "raw_mean":  round(score_pct / 100, 4),
        "status":    "normal" if score_pct >= NORMAL_RANGES["signal_quality_pct"][0] else "low",
    }


# ---------------------------------------------------------------------------
# Waveform data for the frontend chart
# ---------------------------------------------------------------------------

def extract_waveform(
    signal_2d: np.ndarray,
    sampling_rate: int,
    lead_index: int = RHYTHM_LEAD,
    max_samples: int = 20000,
) -> dict:
    """
    Prepare the waveform payload for the frontend chart.

    IMPORTANT: downsampling is for display only. The full-resolution signal
    must be used for peak detection BEFORE calling this function.
    The returned sampling_rate reflects the actual rate of the returned samples.
    """
    lead_names = ["I", "II", "III", "aVR", "aVL", "aVF",
                  "V1", "V2", "V3", "V4", "V5", "V6"]

    lead       = signal_2d[lead_index].astype(float)
    n_samples  = len(lead)
    duration_s = round(n_samples / sampling_rate, 3)

    out_sr = sampling_rate
    if n_samples > max_samples:
        factor = int(np.ceil(n_samples / max_samples))
        trim   = (n_samples // factor) * factor
        lead   = lead[:trim].reshape(-1, factor).mean(axis=1)
        out_sr = sampling_rate // factor

    return {
        "samples":       [round(float(v), 5) for v in lead],
        "sampling_rate": out_sr,
        "duration_s":    duration_s,
        "lead_index":    lead_index,
        "lead_name":     lead_names[lead_index] if lead_index < len(lead_names) else str(lead_index),
    }


# ---------------------------------------------------------------------------
# All-in-one entry point
# ---------------------------------------------------------------------------

def run_full_delineation(
    signal_2d: np.ndarray,
    sampling_rate: int,
) -> dict:
    """
    Run every NeuroKit2 analysis step on the FULL-RESOLUTION signal.

    signal_2d must be (n_leads, n_samples) at the TRUE sampling_rate.
    Downsampling for the waveform payload happens AFTER peak detection.
    """
    # Quality first (doesn't need peaks)
    quality = calculate_quality(signal_2d, sampling_rate)

    # Peak detection on full-resolution signal
    peaks = delineate_peaks(signal_2d, sampling_rate)

    # Intervals from full-resolution peaks
    intervals = calculate_intervals(peaks, sampling_rate)

    # Extract waveform for all available leads
    waveforms = []
    for i in range(signal_2d.shape[0]):
        waveforms.append(extract_waveform(signal_2d, sampling_rate, lead_index=i))

    # Attach status flags
    metrics_with_status = {}
    for key, value in intervals.items():
        if key in NORMAL_RANGES:
            metrics_with_status[key] = {
                "value":  value,
                "status": _status(value, key),
                "unit":   "bpm" if key == "heart_rate_bpm" else "ms",
            }
        else:
            # qt_interval_ms has no normal range defined (QTc is the clinical metric)
            metrics_with_status[key] = {
                "value":  value,
                "status": "info",
                "unit":   "ms",
            }

    return {
        "peaks":     peaks,
        "metrics":   metrics_with_status,
        "quality":   quality,
        "waveforms": waveforms,
    }