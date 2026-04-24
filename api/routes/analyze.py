"""
api/routes/analyze.py

POST /api/v1/analyze/{session_id}
GET  /api/v1/analyze/{session_id}/signal-quality

Changes from the previous version (everything else is identical):
  - _find_hea() is replaced by _load_session(), which auto-detects CSV vs WFDB.
  - loader notes (unit conversion, imputed leads, vendor info) are appended to
    the existing `notes` list and surface in processing_notes as before.
  - The /signal-quality endpoint works for both formats too.
"""

from __future__ import annotations

import logging
import os

from fastapi import APIRouter, HTTPException, Query

from config import settings
import scipy.signal
from core.inference.fcn_wang_runner import ml_pipeline
from core.analysis.delineator import run_full_delineation, calculate_quality
from core.analysis.condition_mapper import map_predictions, get_primary_verdict
from schemas.analysis import (
    AnalysisResponse,
    AnalysisError,
    Verdict,
    MetricsBlock,
    MetricCard,
    WaveformData,
    PeakMarkers,
    WaveAmplitudes,
    AIAnalysis,
    PredictionEntry,
    QualityBadge,
)

from core.loader.ecg_loader import load_session_ecg_with_notes
from core.inference.gradcam import gradcam_runner

router = APIRouter(tags=["Analysis"])
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_session(session_id: str):
    """
    Replaces the old _find_hea() + load_native() pair.

    Returns
    -------
    signal_raw : np.ndarray  (n_samples, 12)  float32  millivolts
    native_sr  : int         sampling rate in Hz
    notes      : list[str]   preprocessing log (unit conversion, imputed leads, …)

    Raises HTTPException 404/422 on missing or unparseable files — same
    behaviour the old _find_hea() had, so callers don't change.
    """
    session_dir = settings.UPLOAD_DIR / session_id

    if not os.path.exists(session_dir):
        raise HTTPException(
            status_code=404,
            detail=f"Session '{session_id}' not found. Upload the file(s) first.",
        )

    # load_session_ecg_with_notes raises HTTPException internally if no valid
    # file is found, so no extra guard needed here.
    signal_raw, native_sr, loader_notes = load_session_ecg_with_notes(str(session_dir))
    return signal_raw, native_sr, loader_notes


def _build_response(
    session_id: str,
    nk_result: dict,
    ai_raw: dict,
    notes: list[str],
) -> AnalysisResponse:
    """Assemble AnalysisResponse from the raw dicts. — UNCHANGED"""

    ai_probs: dict[str, float] = ai_raw.get("probabilities", {})

    # ── AI block ──────────────────────────────────────────────────────────
    top_preds   = map_predictions(ai_probs, top_n=20)
    raw_verdict = get_primary_verdict(ai_probs)

    ai_analysis = AIAnalysis(
        top_predictions=[PredictionEntry(**p) for p in top_preds],
        raw_probabilities={k: round(v, 6) for k, v in ai_probs.items()},
    )

    # ── Quality badge ─────────────────────────────────────────────────────
    quality_badge = QualityBadge(**nk_result["quality"])

    # ── Verdict ───────────────────────────────────────────────────────────
    verdict = Verdict(**raw_verdict, quality=quality_badge)

    # ── Metric cards ──────────────────────────────────────────────────────
    m = nk_result["metrics"]

    def _card(key: str) -> MetricCard:
        entry = m.get(key, {})
        return MetricCard(
            value=entry.get("value"),
            status=entry.get("status", "unknown"),
            unit=entry.get("unit", "ms"),
        )

    metrics = MetricsBlock(
        heart_rate_bpm=_card("heart_rate_bpm"),
        pr_interval_ms=_card("pr_interval_ms"),
        qrs_duration_ms=_card("qrs_duration_ms"),
        qt_interval_ms=_card("qt_interval_ms"),
        qtc_ms=_card("qtc_ms"),
    )

    # ── Waveforms ──────────────────────────────────────────────────────────
    wfs = nk_result.get("waveforms", [])
    waveforms = [WaveformData(**wf) for wf in wfs]

    # ── Peak markers ──────────────────────────────────────────────────────
    pk = nk_result["peaks"]
    peaks = PeakMarkers(
        r_peaks=pk.get("r_peaks", []),
        p_peaks=pk.get("p_peaks", []),
        q_peaks=pk.get("q_peaks", []),
        s_peaks=pk.get("s_peaks", []),
        t_peaks=pk.get("t_peaks", []),
        p_onsets=pk.get("p_onsets", []),
        p_offsets=pk.get("p_offsets", []),
        t_offsets=pk.get("t_offsets", []),
        amplitudes=WaveAmplitudes(**pk.get("amplitudes", {})),
    )

    return AnalysisResponse(
        session_id=session_id,
        status="success" if not notes else "partial",
        verdict=verdict,
        metrics=metrics,
        waveforms=waveforms,
        peaks=peaks,
        ai_analysis=ai_analysis,
        processing_notes=notes,
    )


def _empty_nk_result(signal_2d, sampling_rate: int) -> dict:
    """Zeroed nk_result so _build_response never crashes if delineation fails. — UNCHANGED"""
    n = signal_2d.shape[1] if signal_2d.ndim == 2 else len(signal_2d)
    return {
        "peaks": {
            "r_peaks": [], "p_peaks": [], "q_peaks": [],
            "s_peaks": [], "t_peaks": [], "p_onsets": [],
            "p_offsets": [], "t_offsets": [],
            "amplitudes": {"p": None, "q": None, "r": None, "s": None, "t": None},
        },
        "metrics": {
            "heart_rate_bpm":  {"value": None, "status": "unknown", "unit": "bpm"},
            "pr_interval_ms":  {"value": None, "status": "unknown", "unit": "ms"},
            "qrs_duration_ms": {"value": None, "status": "unknown", "unit": "ms"},
            "qt_interval_ms":  {"value": None, "status": "unknown", "unit": "ms"},
            "qtc_ms":          {"value": None, "status": "unknown", "unit": "ms"},
        },
        "quality": {"score_pct": 0.0, "raw_mean": 0.0, "status": "low"},
        "waveforms": [{
            "samples":       [],
            "sampling_rate": sampling_rate,
            "duration_s":    round(n / sampling_rate, 3),
            "lead_index":    i,
            "lead_name":     str(i),
        } for i in range(12)],
    }


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.post(
    "/analyze/{session_id}",
    response_model=AnalysisResponse,
    responses={
        404: {"model": AnalysisError, "description": "Session not found"},
        422: {"model": AnalysisError, "description": "File could not be parsed"},
        500: {"model": AnalysisError, "description": "Processing error"},
    },
    summary="Full ECG analysis (NeuroKit2 + AI)",
)
async def analyze_ecg(
    session_id: str,
    explain: bool = Query(False, description="Run Grad-CAM beat explainability (CNN)"),
) -> AnalysisResponse:
    notes: list[str] = []

    # ── 1 & 2. Auto-detect format and load ───────────────────────────────
    # Replaces the old: base_path = _find_hea(session_id)
    #                             signal_raw, native_sr = load_native(base_path)
    # Now works for both WFDB pairs AND CSV files transparently.
    try:
        signal_raw, native_sr, loader_notes = _load_session(session_id)
        notes.extend(loader_notes)
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("ECG load failed for session %s", session_id)
        raise HTTPException(status_code=422, detail=f"Could not load ECG file: {exc}")

    # ── 3. NeuroKit2 Pipeline (The Measurer) ─────────────────────────────
    signal_high_res = signal_raw.T  # Transpose to (n_leads, n_samples) — UNCHANGED
    try:
        nk_result = run_full_delineation(signal_high_res, native_sr)
    except Exception as exc:
        logger.exception("NeuroKit2 delineation failed for session %s", session_id)
        notes.append(f"Delineation error (metrics unavailable): {exc}")
        nk_result = _empty_nk_result(signal_high_res, native_sr)

    # ── 4. AI Pipeline (The Diagnoser) ───────────────────────────────────
    try:
        # FCN Wang expects exactly 1000 samples (10s at 100Hz) — UNCHANGED
        signal_ai_input = scipy.signal.resample(signal_raw, 1000)
        ai_raw = ml_pipeline.predict(signal_ai_input)
    except Exception as exc:
        logger.exception("AI inference failed for session %s", session_id)
        notes.append(f"AI inference error (predictions unavailable): {exc}")
        ai_raw = {"probabilities": {}, "detected_conditions": []}

    # ── 5. Grad-CAM (optional) ────────────────────────────────────────────
    gradcam_result = None
    if explain:
        try:
            lead2      = signal_raw[:, 1] if signal_raw.shape[1] > 1 else signal_raw[:, 0]
            r_peaks_nk = nk_result["peaks"].get("r_peaks", [])
            gradcam_result = gradcam_runner.run(lead2, native_sr, r_peaks_nk)
        except Exception as exc:
            logger.warning("Grad-CAM failed for session %s: %s", session_id, exc)
            notes.append(f"Grad-CAM unavailable: {exc}")

    # ── 6. Assemble and return ────────────────────────────────────────────
    try:
        response = _build_response(session_id, nk_result, ai_raw, notes)
        if gradcam_result is not None:
            response.gradcam = gradcam_result
        return response
    except Exception as exc:
        logger.exception("Response assembly failed for session %s", session_id)
        raise HTTPException(status_code=500, detail=f"Response assembly error: {exc}")


@router.get(
    "/analyze/{session_id}/signal-quality",
    summary="Signal quality check only",
)
async def signal_quality(session_id: str) -> dict:
    """Works for both WFDB and CSV sessions — UNCHANGED from caller's perspective."""
    try:
        signal_raw, native_sr, _ = _load_session(session_id)
        signal_high_res = signal_raw.T
        quality = calculate_quality(signal_high_res, native_sr)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    return {"session_id": session_id, "signal_quality": quality}