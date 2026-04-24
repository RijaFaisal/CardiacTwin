"""
api/routes/export.py

POST /api/v1/export/{session_id}/pdf

Accepts the AnalysisResponse JSON the frontend already holds and returns
a formatted PDF clinical report as a file download.

Dependencies:  pip install reportlab
"""

from __future__ import annotations

import io
import logging
from datetime import datetime

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from schemas.analysis import AnalysisResponse

router = APIRouter(tags=["Export"])
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# PDF builder
# ---------------------------------------------------------------------------

def _build_pdf(data: AnalysisResponse) -> bytes:
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib import colors
        from reportlab.lib.units import mm
        from reportlab.platypus import (
            SimpleDocTemplate, Paragraph, Spacer,
            Table, TableStyle, HRFlowable,
        )
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.enums import TA_CENTER
    except ImportError as exc:
        raise RuntimeError("reportlab is not installed. Run: pip install reportlab") from exc

    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        rightMargin=20*mm, leftMargin=20*mm,
        topMargin=20*mm,   bottomMargin=20*mm,
    )
    styles  = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "Title", parent=styles["Heading1"],
        fontSize=18, spaceAfter=4, alignment=TA_CENTER,
    )
    section_style = ParagraphStyle(
        "Section", parent=styles["Heading2"],
        fontSize=12, spaceBefore=10, spaceAfter=4,
        textColor=colors.HexColor("#1a1a2e"),
    )
    normal_style = styles["Normal"]
    small_style  = ParagraphStyle(
        "Small", parent=styles["Normal"],
        fontSize=8, textColor=colors.grey,
    )

    sev_colours = {
        "normal":   colors.HexColor("#16a34a"),
        "warning":  colors.HexColor("#d97706"),
        "critical": colors.HexColor("#dc2626"),
    }

    verdict   = data.verdict
    sev_color = sev_colours.get(verdict.severity, colors.grey)
    story     = []

    # ── Header ────────────────────────────────────────────────────────────
    story.append(Paragraph("ECG Analysis Report", title_style))
    story.append(Paragraph(
        f"Session ID: <b>{data.session_id}</b> &nbsp;|&nbsp; "
        f"Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}",
        normal_style,
    ))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.lightgrey, spaceAfter=8))

    # ── Verdict ───────────────────────────────────────────────────────────
    story.append(Paragraph("Primary Diagnosis", section_style))
    verdict_table = Table(
        [["Finding", "Confidence", "Severity"],
         [verdict.display_name, f"{verdict.percentage}%", verdict.severity.upper()]],
        colWidths=["60%", "20%", "20%"],
    )
    verdict_table.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, 0), colors.HexColor("#1a1a2e")),
        ("TEXTCOLOR",     (0, 0), (-1, 0), colors.white),
        ("FONTNAME",      (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",      (0, 0), (-1, -1), 10),
        ("GRID",          (0, 0), (-1, -1), 0.5, colors.lightgrey),
        ("ROWBACKGROUNDS",(0, 1), (-1, -1), [colors.HexColor("#f9f9f9")]),
        ("TEXTCOLOR",     (2, 1), (2, 1),   sev_color),
        ("FONTNAME",      (2, 1), (2, 1),   "Helvetica-Bold"),
        ("ALIGN",         (1, 0), (-1, -1), "CENTER"),
        ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING",    (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    story.append(verdict_table)
    story.append(Paragraph(verdict.definition, small_style))
    story.append(Spacer(1, 6))

    # ── Metrics ───────────────────────────────────────────────────────────
    story.append(Paragraph("Clinical Measurements", section_style))
    m = data.metrics

    def _fmt(card) -> str:
        return f"{round(card.value, 1)} {card.unit}" if card.value is not None else "N/A"

    def _status_txt(card) -> str:
        return card.status.upper()

    metrics_table = Table(
        [["Measurement",   "Value",              "Status"],
         ["Heart Rate",    _fmt(m.heart_rate_bpm),  _status_txt(m.heart_rate_bpm)],
         ["PR Interval",   _fmt(m.pr_interval_ms),  _status_txt(m.pr_interval_ms)],
         ["QRS Duration",  _fmt(m.qrs_duration_ms), _status_txt(m.qrs_duration_ms)],
         ["QT Interval",   _fmt(m.qt_interval_ms),  _status_txt(m.qt_interval_ms)],
         ["QTc (Bazett)",  _fmt(m.qtc_ms),           _status_txt(m.qtc_ms)]],
        colWidths=["50%", "25%", "25%"],
    )
    metrics_table.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, 0), colors.HexColor("#374151")),
        ("TEXTCOLOR",     (0, 0), (-1, 0), colors.white),
        ("FONTNAME",      (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",      (0, 0), (-1, -1), 10),
        ("GRID",          (0, 0), (-1, -1), 0.5, colors.lightgrey),
        ("ROWBACKGROUNDS",(0, 1), (-1, -1), [colors.HexColor("#f9f9f9"), colors.white]),
        ("ALIGN",         (1, 0), (-1, -1), "CENTER"),
        ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING",    (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    story.append(metrics_table)
    story.append(Spacer(1, 6))

    # ── AI Predictions ────────────────────────────────────────────────────
    story.append(Paragraph("AI Diagnostic Analysis (Top 3)", section_style))
    ai_rows = [["Condition", "Code", "Confidence", "Severity"]]
    for pred in data.ai_analysis.top_predictions:
        ai_rows.append([
            pred.display_name,
            pred.code,
            f"{pred.percentage}%",
            pred.severity.upper(),
        ])
    ai_table = Table(ai_rows, colWidths=["45%", "15%", "20%", "20%"])
    ai_table.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, 0), colors.HexColor("#374151")),
        ("TEXTCOLOR",     (0, 0), (-1, 0), colors.white),
        ("FONTNAME",      (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",      (0, 0), (-1, -1), 10),
        ("GRID",          (0, 0), (-1, -1), 0.5, colors.lightgrey),
        ("ROWBACKGROUNDS",(0, 1), (-1, -1), [colors.HexColor("#f9f9f9"), colors.white]),
        ("ALIGN",         (1, 0), (-1, -1), "CENTER"),
        ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING",    (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    story.append(ai_table)
    story.append(Spacer(1, 6))

    # ── Signal quality ────────────────────────────────────────────────────
    story.append(Paragraph("Signal Quality", section_style))
    q = verdict.quality
    story.append(Paragraph(
        f"Quality Score: <b>{q.score_pct}%</b> — {q.status.upper()}",
        normal_style,
    ))
    story.append(Spacer(1, 4))

    # ── Processing notes ──────────────────────────────────────────────────
    if data.processing_notes:
        story.append(Paragraph("Processing Notes", section_style))
        for note in data.processing_notes:
            story.append(Paragraph(f"• {note}", small_style))
        story.append(Spacer(1, 4))

    # ── Disclaimer ────────────────────────────────────────────────────────
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.lightgrey, spaceBefore=10))
    story.append(Paragraph(
        "DISCLAIMER: This report is generated by an AI-assisted ECG analysis system and is "
        "intended for informational purposes only. It does not constitute a medical diagnosis. "
        "All findings must be reviewed and interpreted by a qualified healthcare professional.",
        small_style,
    ))

    doc.build(story)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Route
# ---------------------------------------------------------------------------

@router.post(
    "/export/{session_id}/pdf",
    summary="Export analysis as PDF clinical report",
)
async def export_pdf(session_id: str, analysis: AnalysisResponse) -> StreamingResponse:
    try:
        pdf_bytes = _build_pdf(analysis)
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    except Exception as exc:
        logger.exception("PDF generation failed for session %s", session_id)
        raise HTTPException(status_code=500, detail=f"PDF generation error: {exc}")

    return StreamingResponse(
        io.BytesIO(pdf_bytes),
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="ecg_report_{session_id}.pdf"'},
    )