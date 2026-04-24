"""
schemas/analysis.py

Pydantic models for the unified POST /api/analyze/{session_id} response.

Every field name is chosen to map directly to a frontend panel:
  AnalysisResponse.verdict      → Header / badge
  AnalysisResponse.metrics      → Left pane metric cards
  AnalysisResponse.waveform     → Center chart
  AnalysisResponse.ai_analysis  → Right pane bar chart

Keep this file in sync with core/analysis/delineator.py and
core/analysis/condition_mapper.py.
"""

from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Shared primitives
# ---------------------------------------------------------------------------

class MetricCard(BaseModel):
    """A single measurement displayed as a card in the left pane."""
    value:  Optional[float] = Field(None, description="Measured value (None if could not be computed)")
    status: str             = Field(...,  description="'normal' | 'low' | 'high' | 'unknown'")
    unit:   str             = Field(...,  description="Display unit, e.g. 'bpm' or 'ms'")


class WaveAmplitudes(BaseModel):
    """Mean amplitude (in signal units) per wave type for the rhythm lead."""
    p: Optional[float] = None
    q: Optional[float] = None
    r: Optional[float] = None
    s: Optional[float] = None
    t: Optional[float] = None


class PeakMarkers(BaseModel):
    """
    Sample-index arrays for each wave type, ready for the frontend to
    overlay as vertical markers on the waveform chart.
    """
    r_peaks:   list[int] = Field(default_factory=list)
    p_peaks:   list[int] = Field(default_factory=list)
    q_peaks:   list[int] = Field(default_factory=list)
    s_peaks:   list[int] = Field(default_factory=list)
    t_peaks:   list[int] = Field(default_factory=list)
    p_onsets:  list[int] = Field(default_factory=list)
    p_offsets: list[int] = Field(default_factory=list)
    t_offsets: list[int] = Field(default_factory=list)
    amplitudes: WaveAmplitudes = Field(default_factory=WaveAmplitudes)


class WaveformData(BaseModel):
    """
    Centre-pane waveform payload.

    `samples` is a downsampled amplitude array at `sampling_rate` Hz.
    The frontend can map index → time_s = index / sampling_rate.
    """
    samples:       list[float]
    sampling_rate: int
    duration_s:    float
    lead_index:    int
    lead_name:     str


class QualityBadge(BaseModel):
    """Signal integrity score shown as a badge in the dashboard header."""
    score_pct: float = Field(..., ge=0, le=100, description="Quality percentage 0–100")
    raw_mean:  float
    status:    str   = Field(..., description="'normal' | 'low'")


# ---------------------------------------------------------------------------
# AI analysis block (right pane)
# ---------------------------------------------------------------------------

class PredictionEntry(BaseModel):
    """A single classification result in the right-pane bar chart."""
    code:         str
    display_name: str
    definition:   str   = Field(..., description="Tooltip text for the '?' icon")
    probability:  float = Field(..., ge=0.0, le=1.0)
    percentage:   int   = Field(..., ge=0,   le=100)
    severity:     str   = Field(..., description="'normal' | 'warning' | 'critical'")


class AIAnalysis(BaseModel):
    """Right pane: top-N predictions + the full raw probability map."""
    top_predictions:   list[PredictionEntry]
    raw_probabilities: dict[str, float] = Field(
        default_factory=dict,
        description="Full 71-class probability map from the model",
    )


# ---------------------------------------------------------------------------
# Verdict block (header)
# ---------------------------------------------------------------------------

class Verdict(BaseModel):
    """
    Primary headline diagnosis shown in the dashboard header.

    severity drives the colour coding:
      normal   → green
      warning  → yellow
      critical → red
    """
    code:            str
    display_name:    str
    definition:      str
    probability:     float
    percentage:      int
    severity:        str
    above_threshold: bool
    quality:         QualityBadge


# ---------------------------------------------------------------------------
# Metrics block (left pane)
# ---------------------------------------------------------------------------

class MetricsBlock(BaseModel):
    """Left-pane metric cards. Each field is one card."""
    heart_rate_bpm:  MetricCard
    pr_interval_ms:  MetricCard
    qrs_duration_ms: MetricCard
    qt_interval_ms:  MetricCard
    qtc_ms:          MetricCard


# ---------------------------------------------------------------------------
# Root response model
# ---------------------------------------------------------------------------

class AnalysisResponse(BaseModel):
    """
    Unified response from POST /api/analyze/{session_id}.

    Designed so each top-level key maps 1-to-1 to a frontend panel.
    """
    session_id:  str
    status:      str = Field("success", description="'success' | 'partial' | 'error'")

    # ── Dashboard panels ───────────────────────────────────────────────────
    verdict:     Verdict
    metrics:     MetricsBlock
    waveforms:   list[WaveformData]
    peaks:       PeakMarkers
    ai_analysis: AIAnalysis

    # ── Explainability (present only when ?explain=true) ──────────────────
    gradcam: Optional[dict] = Field(
        None,
        description=(
            "Grad-CAM explainability. "
            "gradcam.fcn_wang: temporal saliency map from FCN Wang (explains the actual diagnosis). "
            "gradcam.mitbih_cnn: per-beat saliency from MIT-BIH CNN (beat morphology detail)."
        ),
    )

    # ── Metadata ──────────────────────────────────────────────────────────
    processing_notes: list[str] = Field(
        default_factory=list,
        description="Non-fatal warnings (e.g. 'Lead II poor quality, used Lead I')",
    )


# ---------------------------------------------------------------------------
# Error response (returned when analysis fails entirely)
# ---------------------------------------------------------------------------

class AnalysisError(BaseModel):
    session_id: str
    status:     str = "error"
    detail:     str