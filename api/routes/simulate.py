"""
api/routes/simulate.py

POST /api/v1/simulate/pathology          — hardcoded pathology demo (kept for UI)
POST /api/v1/simulate/treatment          — hardcoded treatment demo (kept for UI)
GET  /api/v1/simulate/from-session/{id} — ML-derived simulation from FCN Wang output
GET  /api/v1/simulate/metrics           — model evaluation metrics
"""

import json
import os

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from storage.session_store import load as load_session

router = APIRouter(prefix="/api/v1/simulate", tags=["simulate"])


# ── Request models ─────────────────────────────────────────────────────────

class PathologyRequest(BaseModel):
    pathology: str
    age: int = 45
    gender: str = "M"


class TreatmentRequest(BaseModel):
    treatment: str
    pathology: str = "none"
    age: int = 45
    gender: str = "M"


# ── Static demo tables (kept for the manual simulation UI panel) ───────────

_PATHOLOGY_PARAMS = {
    "none": {
        "display_name":      "Normal Sinus Rhythm",
        "heart_rate":        72.0,
        "hrv_sdnn":          0.045,
        "hrv_rmssd":         0.038,
        "rr_irregular":      False,
        "dominant_class":    "Normal",
        "beat_distribution": {"Normal": 95.0, "Bundle Branch Block": 1.0,
                               "Ventricular": 2.0, "Atrial": 1.0, "Other": 1.0},
        "description": "Regular P-QRS-T complexes. Normal conduction velocity.",
    },
    "af": {
        "display_name":      "Atrial Fibrillation",
        "heart_rate":        110.0,
        "hrv_sdnn":          0.185,
        "hrv_rmssd":         0.210,
        "rr_irregular":      True,
        "dominant_class":    "Atrial",
        "beat_distribution": {"Normal": 12.0, "Atrial": 80.0,
                               "Ventricular": 5.0, "Bundle Branch Block": 1.0, "Other": 2.0},
        "description": "Absent P-waves, irregularly irregular RR intervals. "
                       "Rapid chaotic atrial activity at 350-600 impulses/min.",
    },
    "mi": {
        "display_name":      "Myocardial Infarction",
        "heart_rate":        88.0,
        "hrv_sdnn":          0.028,
        "hrv_rmssd":         0.021,
        "rr_irregular":      False,
        "dominant_class":    "Other",
        "beat_distribution": {"Normal": 40.0, "Ventricular": 20.0,
                               "Other": 35.0, "Atrial": 3.0, "Bundle Branch Block": 2.0},
        "description": "ST-segment elevation with Q-wave formation. "
                       "Ischaemic zone reduces contractility and wall motion.",
    },
    "vt": {
        "display_name":      "Ventricular Tachycardia",
        "heart_rate":        165.0,
        "hrv_sdnn":          0.012,
        "hrv_rmssd":         0.009,
        "rr_irregular":      False,
        "dominant_class":    "Ventricular",
        "beat_distribution": {"Ventricular": 92.0, "Normal": 5.0,
                               "Other": 2.0, "Atrial": 1.0, "Bundle Branch Block": 0.0},
        "description": "Wide-complex tachycardia originating in ventricular myocardium. "
                       "Haemodynamically unstable if sustained.",
    },
    "hypertrophy": {
        "display_name":      "Ventricular Hypertrophy",
        "heart_rate":        62.0,
        "hrv_sdnn":          0.055,
        "hrv_rmssd":         0.042,
        "rr_irregular":      False,
        "dominant_class":    "Other",
        "beat_distribution": {"Normal": 55.0, "Other": 35.0,
                               "Bundle Branch Block": 5.0, "Ventricular": 3.0, "Atrial": 2.0},
        "description": "Increased ventricular wall thickness. "
                       "High-voltage QRS with repolarisation strain pattern.",
    },
}

_TREATMENT_PARAMS = {
    "pacemaker": {
        "display_name":      "Pacemaker Implantation",
        "heart_rate":        70.0,
        "hrv_sdnn":          0.015,
        "hrv_rmssd":         0.010,
        "rr_irregular":      False,
        "dominant_class":    "Normal",
        "beat_distribution": {"Normal": 95.0, "Ventricular": 3.0,
                               "Other": 1.0, "Atrial": 1.0, "Bundle Branch Block": 0.0},
        "description": "Electrical pacing at 70 BPM. Regular ventricular activation restored.",
        "efficacy":    "Rhythm regularized. Rate controlled at 70 BPM.",
    },
    "medication": {
        "display_name":      "Antiarrhythmic Medication",
        "heart_rate":        72.0,
        "hrv_sdnn":          0.040,
        "hrv_rmssd":         0.035,
        "rr_irregular":      False,
        "dominant_class":    "Normal",
        "beat_distribution": {"Normal": 85.0, "Atrial": 8.0,
                               "Ventricular": 4.0, "Other": 2.0, "Bundle Branch Block": 1.0},
        "description": "Pharmacological rate and rhythm control. "
                       "Partial normalization of conduction.",
        "efficacy":    "Heart rate reduced. Rhythm partially regularized.",
    },
    "ablation": {
        "display_name":      "Catheter Ablation",
        "heart_rate":        68.0,
        "hrv_sdnn":          0.048,
        "hrv_rmssd":         0.040,
        "rr_irregular":      False,
        "dominant_class":    "Normal",
        "beat_distribution": {"Normal": 92.0, "Other": 4.0,
                               "Ventricular": 2.0, "Atrial": 1.0, "Bundle Branch Block": 1.0},
        "description": "Ablation of ectopic foci. Accessory conduction pathways eliminated.",
        "efficacy":    "Arrhythmia source eliminated. Sinus rhythm restored.",
    },
}


# ── ML-derived simulation: physiological parameter contributions ───────────
#
# Each PTB-XL SCP code contributes delta values from the normal baseline:
#   heart_rate_delta  : bpm above/below 72 bpm
#   hrv_sdnn_delta    : seconds above/below 0.045 s
#   hrv_rmssd_delta   : seconds above/below 0.038 s
#   irregular_weight  : 0–1; contributes to RR irregularity determination
#   severity_weight   : used to set dominant condition in output
#
# These are not hardcoded outputs — they are contributions blended by the
# FCN Wang probability for each code, producing parameters unique to each ECG.

_CODE_PHYSIOLOGY: dict[str, dict] = {
    # ── Normal / sinus ────────────────────────────────────────────────────
    "NORM":  {"hr": 0,    "sdnn": 0,      "rmssd": 0,      "irr": 0.0, "sev": 0.0},
    "SR":    {"hr": 0,    "sdnn": 0,      "rmssd": 0,      "irr": 0.0, "sev": 0.0},
    "SARRH": {"hr": 0,    "sdnn":+0.005,  "rmssd":+0.004,  "irr": 0.1, "sev": 0.1},
    "PACE":  {"hr": 0,    "sdnn":-0.030,  "rmssd":-0.028,  "irr": 0.0, "sev": 0.2},
    # ── Sinus rate abnormalities ──────────────────────────────────────────
    "STACH": {"hr":+30,   "sdnn":-0.010,  "rmssd":-0.008,  "irr": 0.0, "sev": 0.4},
    "SBRAD": {"hr":-22,   "sdnn":+0.008,  "rmssd":+0.005,  "irr": 0.0, "sev": 0.4},
    # ── Atrial arrhythmias ────────────────────────────────────────────────
    "AFIB":  {"hr":+38,   "sdnn":+0.140,  "rmssd":+0.172,  "irr": 1.0, "sev": 1.0},
    "AFLT":  {"hr":+38,   "sdnn":+0.050,  "rmssd":+0.060,  "irr": 0.6, "sev": 0.9},
    "PAC":   {"hr": 0,    "sdnn":+0.012,  "rmssd":+0.010,  "irr": 0.3, "sev": 0.3},
    "PSVT":  {"hr":+60,   "sdnn":-0.015,  "rmssd":-0.012,  "irr": 0.2, "sev": 0.5},
    "SVARR": {"hr":+10,   "sdnn":+0.020,  "rmssd":+0.015,  "irr": 0.4, "sev": 0.5},
    "SVTAC": {"hr":+50,   "sdnn":-0.012,  "rmssd":-0.010,  "irr": 0.2, "sev": 0.6},
    # ── Ventricular ───────────────────────────────────────────────────────
    "PVC":   {"hr": 0,    "sdnn":+0.025,  "rmssd":+0.020,  "irr": 0.5, "sev": 0.5},
    "BIGU":  {"hr": 0,    "sdnn":+0.020,  "rmssd":+0.015,  "irr": 0.6, "sev": 0.5},
    "TRIGU": {"hr": 0,    "sdnn":+0.015,  "rmssd":+0.010,  "irr": 0.4, "sev": 0.4},
    # ── Infarction ────────────────────────────────────────────────────────
    "AMI":   {"hr":+16,   "sdnn":-0.017,  "rmssd":-0.017,  "irr": 0.1, "sev": 1.0},
    "IMI":   {"hr":+10,   "sdnn":-0.012,  "rmssd":-0.010,  "irr": 0.1, "sev": 0.9},
    "ASMI":  {"hr":+12,   "sdnn":-0.014,  "rmssd":-0.012,  "irr": 0.1, "sev": 0.9},
    "ALMI":  {"hr":+12,   "sdnn":-0.014,  "rmssd":-0.012,  "irr": 0.1, "sev": 0.9},
    "ILMI":  {"hr":+10,   "sdnn":-0.012,  "rmssd":-0.010,  "irr": 0.1, "sev": 0.8},
    "LMI":   {"hr":+10,   "sdnn":-0.011,  "rmssd":-0.009,  "irr": 0.1, "sev": 0.8},
    "IPLMI": {"hr":+11,   "sdnn":-0.013,  "rmssd":-0.011,  "irr": 0.1, "sev": 0.9},
    "IPMI":  {"hr":+11,   "sdnn":-0.013,  "rmssd":-0.011,  "irr": 0.1, "sev": 0.9},
    "PMI":   {"hr":+10,   "sdnn":-0.012,  "rmssd":-0.010,  "irr": 0.1, "sev": 0.8},
    # ── Ischaemia ─────────────────────────────────────────────────────────
    "ISCAL": {"hr":+8,    "sdnn":-0.008,  "rmssd":-0.007,  "irr": 0.1, "sev": 0.8},
    "ISCAN": {"hr":+8,    "sdnn":-0.008,  "rmssd":-0.007,  "irr": 0.1, "sev": 0.8},
    "ISCAS": {"hr":+8,    "sdnn":-0.008,  "rmssd":-0.007,  "irr": 0.1, "sev": 0.8},
    "ISCIL": {"hr":+8,    "sdnn":-0.008,  "rmssd":-0.007,  "irr": 0.1, "sev": 0.7},
    "ISCIN": {"hr":+8,    "sdnn":-0.008,  "rmssd":-0.007,  "irr": 0.1, "sev": 0.7},
    "ISCLA": {"hr":+8,    "sdnn":-0.008,  "rmssd":-0.007,  "irr": 0.1, "sev": 0.7},
    "ISC_":  {"hr":+5,    "sdnn":-0.005,  "rmssd":-0.004,  "irr": 0.1, "sev": 0.5},
    # ── Bundle branch blocks ──────────────────────────────────────────────
    "CLBBB": {"hr":-3,    "sdnn":-0.005,  "rmssd":-0.003,  "irr": 0.0, "sev": 0.5},
    "CRBBB": {"hr":-2,    "sdnn":-0.003,  "rmssd":-0.002,  "irr": 0.0, "sev": 0.4},
    "ILBBB": {"hr":-2,    "sdnn":-0.003,  "rmssd":-0.002,  "irr": 0.0, "sev": 0.3},
    "IRBBB": {"hr":-1,    "sdnn":-0.002,  "rmssd":-0.001,  "irr": 0.0, "sev": 0.2},
    "IVCD":  {"hr": 0,    "sdnn":-0.003,  "rmssd":-0.002,  "irr": 0.0, "sev": 0.3},
    "LAFB":  {"hr": 0,    "sdnn":-0.002,  "rmssd":-0.001,  "irr": 0.0, "sev": 0.2},
    "LPFB":  {"hr": 0,    "sdnn":-0.002,  "rmssd":-0.001,  "irr": 0.0, "sev": 0.2},
    # ── Hypertrophy ───────────────────────────────────────────────────────
    "LVH":   {"hr":-8,    "sdnn":-0.010,  "rmssd":-0.008,  "irr": 0.0, "sev": 0.5},
    "RVH":   {"hr":-5,    "sdnn":-0.007,  "rmssd":-0.005,  "irr": 0.0, "sev": 0.5},
    "SEHYP": {"hr":-4,    "sdnn":-0.005,  "rmssd":-0.004,  "irr": 0.0, "sev": 0.4},
    # ── AV blocks ─────────────────────────────────────────────────────────
    "1AVB":  {"hr":-3,    "sdnn":+0.003,  "rmssd":+0.002,  "irr": 0.0, "sev": 0.4},
    "2AVB":  {"hr":-12,   "sdnn":+0.010,  "rmssd":+0.008,  "irr": 0.3, "sev": 0.8},
    "3AVB":  {"hr":-30,   "sdnn":+0.020,  "rmssd":+0.015,  "irr": 0.0, "sev": 1.0},
    "WPW":   {"hr":+15,   "sdnn":-0.008,  "rmssd":-0.006,  "irr": 0.3, "sev": 0.8},
    "LNGQT": {"hr": 0,    "sdnn":-0.005,  "rmssd":-0.003,  "irr": 0.1, "sev": 0.7},
    # ── ST-T / injury ─────────────────────────────────────────────────────
    "STE_":  {"hr":+10,   "sdnn":-0.010,  "rmssd":-0.008,  "irr": 0.1, "sev": 0.9},
    "STD_":  {"hr":+6,    "sdnn":-0.007,  "rmssd":-0.005,  "irr": 0.1, "sev": 0.6},
    "INJAL": {"hr":+10,   "sdnn":-0.010,  "rmssd":-0.008,  "irr": 0.1, "sev": 0.9},
    "INJAS": {"hr":+10,   "sdnn":-0.010,  "rmssd":-0.008,  "irr": 0.1, "sev": 0.9},
    "INJIL": {"hr":+10,   "sdnn":-0.010,  "rmssd":-0.008,  "irr": 0.1, "sev": 0.9},
    "INJIN": {"hr":+10,   "sdnn":-0.010,  "rmssd":-0.008,  "irr": 0.1, "sev": 0.9},
    "INJLA": {"hr":+10,   "sdnn":-0.010,  "rmssd":-0.008,  "irr": 0.1, "sev": 0.9},
}

_BASE_HR    = 72.0
_BASE_SDNN  = 0.045
_BASE_RMSSD = 0.038
_THRESHOLD  = 0.30   # minimum probability to include a condition's contribution


def _derive_simulation_params(probabilities: dict[str, float]) -> dict:
    """
    Derive cardiac simulation parameters directly from FCN Wang probabilities.

    Each detected condition (prob >= threshold) contributes weighted deltas
    to the baseline normal sinus rhythm parameters.  The result is unique
    to the uploaded ECG — not a lookup table selection.
    """
    detected = {
        code: prob
        for code, prob in probabilities.items()
        if prob >= _THRESHOLD and code in _CODE_PHYSIOLOGY
    }

    if not detected:
        # Nothing above threshold — return normal baseline
        return {
            "source":         "ml_derived",
            "display_name":   "Normal / No significant findings",
            "heart_rate":     _BASE_HR,
            "hrv_sdnn":       _BASE_SDNN,
            "hrv_rmssd":      _BASE_RMSSD,
            "rr_irregular":   False,
            "detected_codes": [],
            "description":    "No conditions above detection threshold. "
                              "Parameters reflect normal sinus rhythm.",
        }

    total_prob = sum(detected.values())

    hr_delta    = sum(_CODE_PHYSIOLOGY[c]["hr"]    * p for c, p in detected.items()) / total_prob
    sdnn_delta  = sum(_CODE_PHYSIOLOGY[c]["sdnn"]  * p for c, p in detected.items()) / total_prob
    rmssd_delta = sum(_CODE_PHYSIOLOGY[c]["rmssd"] * p for c, p in detected.items()) / total_prob
    irr_score   = sum(_CODE_PHYSIOLOGY[c]["irr"]   * p for c, p in detected.items()) / total_prob

    # Dominant condition by probability × severity weight
    dominant_code = max(
        detected,
        key=lambda c: detected[c] * _CODE_PHYSIOLOGY[c]["sev"]
    )

    # Condition descriptions for context
    detected_list = [
        {"code": c, "probability": round(p, 3)}
        for c, p in sorted(detected.items(), key=lambda x: -x[1])
    ]

    return {
        "source":         "ml_derived",
        "display_name":   f"ML-derived ({dominant_code} dominant)",
        "heart_rate":     round(max(20.0, min(220.0, _BASE_HR + hr_delta)), 1),
        "hrv_sdnn":       round(max(0.005, _BASE_SDNN + sdnn_delta), 3),
        "hrv_rmssd":      round(max(0.003, _BASE_RMSSD + rmssd_delta), 3),
        "rr_irregular":   irr_score >= 0.40,
        "dominant_code":  dominant_code,
        "detected_codes": detected_list,
        "description": (
            f"Parameters derived from FCN Wang predictions on this ECG. "
            f"Dominant finding: {dominant_code} "
            f"({round(probabilities.get(dominant_code, 0) * 100)}% probability). "
            f"{len(detected)} condition(s) above {int(_THRESHOLD * 100)}% threshold."
        ),
    }


def _apply_demographics(params: dict, age: int, gender: str) -> dict:
    result = dict(params)
    hr = result["heart_rate"]
    if age < 30:
        hr *= 1.05
    elif age > 60:
        hr *= 0.93
    if gender.upper() == "F":
        hr *= 1.03
    result["heart_rate"]       = round(hr, 1)
    result["demographic_note"] = (
        f"Age {age}, {'Female' if gender.upper() == 'F' else 'Male'}"
    )
    return result


# ── Endpoints ──────────────────────────────────────────────────────────────

@router.post("/pathology")
def simulate_pathology(req: PathologyRequest):
    params = _PATHOLOGY_PARAMS.get(req.pathology)
    if params is None:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown pathology '{req.pathology}'. "
                   f"Valid values: {list(_PATHOLOGY_PARAMS)}",
        )
    return _apply_demographics(dict(params), req.age, req.gender)


@router.post("/treatment")
def simulate_treatment(req: TreatmentRequest):
    treatment = _TREATMENT_PARAMS.get(req.treatment)
    if treatment is None:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown treatment '{req.treatment}'. "
                   f"Valid values: {list(_TREATMENT_PARAMS)}",
        )
    pathology = _PATHOLOGY_PARAMS.get(req.pathology, _PATHOLOGY_PARAMS["none"])
    result = dict(treatment)
    result["pathology_before"]  = req.pathology
    result["pathology_display"] = pathology["display_name"]
    return _apply_demographics(result, req.age, req.gender)


@router.get("/from-session/{session_id}")
def simulate_from_session(session_id: str):
    """
    Derive cardiac simulation parameters directly from the FCN Wang
    predictions stored for this session.

    Unlike /pathology (which uses a fixed lookup table), these parameters
    are computed from the actual probability distribution over 71 conditions
    detected in the uploaded ECG — making the digital twin reflect this
    specific patient's ECG.
    """
    session = load_session(session_id)
    if session is None:
        raise HTTPException(
            status_code=404,
            detail=f"Session '{session_id}' not found or not yet analyzed. "
                   "Run /analyze first.",
        )

    probs = (
        session.get("ai_analysis", {}).get("raw_probabilities", {})
    )
    if not probs:
        raise HTTPException(
            status_code=422,
            detail="No AI predictions found for this session. "
                   "Re-run /analyze to generate predictions.",
        )

    return _derive_simulation_params(probs)


@router.get("/metrics")
def get_model_metrics():
    """Return evaluation metrics for all models (FCN Wang + MIT-BIH CNN + baselines)."""
    fcn_wang_path = "ml/results/fcn_wang_metrics.json"
    cnn_path      = "ml/results/cnn_metrics.json"
    baseline_path = "ml/results/baseline_metrics.json"

    fcn_wang = None
    cnn      = None
    baseline = None

    if os.path.exists(fcn_wang_path):
        with open(fcn_wang_path) as f:
            fcn_wang = json.load(f)
    if os.path.exists(cnn_path):
        with open(cnn_path) as f:
            cnn = json.load(f)
    if os.path.exists(baseline_path):
        with open(baseline_path) as f:
            baseline = json.load(f)

    if fcn_wang is None and cnn is None and baseline is None:
        raise HTTPException(
            status_code=404,
            detail="No metrics found. Run ml/evaluate_fcn_wang.py and ml/evaluate.py first.",
        )

    return {
        "production_model": fcn_wang,   # FCN Wang / PTB-XL — primary model
        "cnn_mitbih":       cnn,         # MIT-BIH CNN — beat-level baseline
        "baselines":        baseline,    # SVM + Logistic Regression
    }
