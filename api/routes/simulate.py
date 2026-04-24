"""
api/routes/simulate.py

POST /api/v1/simulate/pathology  — return simulated cardiac params for a pathology
POST /api/v1/simulate/treatment  — return simulated post-treatment cardiac params
GET  /api/v1/simulate/metrics    — return CNN vs baseline evaluation metrics
"""

import json
import os

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter(prefix="/api/v1/simulate", tags=["simulate"])


# ── Request models ─────────────────────────────────────────────────

class PathologyRequest(BaseModel):
    pathology: str
    age: int = 45
    gender: str = "M"


class TreatmentRequest(BaseModel):
    treatment: str
    pathology: str = "none"
    age: int = 45
    gender: str = "M"


# ── Pathology parameter table ──────────────────────────────────────

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
        "description": "Increased ventricular wall thickness. High-voltage QRS with repolarisation strain pattern.",
    },
}

# ── Treatment parameter table ──────────────────────────────────────

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
        "description": "Pharmacological rate and rhythm control. Partial normalization of conduction.",
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


def _apply_demographics(params: dict, age: int, gender: str) -> dict:
    result = dict(params)
    hr = result["heart_rate"]
    if age < 30:
        hr *= 1.05
    elif age > 60:
        hr *= 0.93
    if gender.upper() == "F":
        hr *= 1.03
    result["heart_rate"] = round(hr, 1)
    result["demographic_note"] = (
        f"Age {age}, {'Female' if gender.upper() == 'F' else 'Male'}"
    )
    return result


# ── Endpoints ─────────────────────────────────────────────────────

@router.post("/pathology")
def simulate_pathology(req: PathologyRequest):
    params = _PATHOLOGY_PARAMS.get(req.pathology)
    if params is None:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown pathology '{req.pathology}'. "
                   f"Valid values: {list(_PATHOLOGY_PARAMS)}"
        )
    return _apply_demographics(dict(params), req.age, req.gender)


@router.post("/treatment")
def simulate_treatment(req: TreatmentRequest):
    treatment = _TREATMENT_PARAMS.get(req.treatment)
    if treatment is None:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown treatment '{req.treatment}'. "
                   f"Valid values: {list(_TREATMENT_PARAMS)}"
        )
    pathology = _PATHOLOGY_PARAMS.get(req.pathology, _PATHOLOGY_PARAMS["none"])
    result = dict(treatment)
    result["pathology_before"]  = req.pathology
    result["pathology_display"] = pathology["display_name"]
    result = _apply_demographics(result, req.age, req.gender)
    return result


@router.get("/metrics")
def get_model_metrics():
    cnn_path      = "ml/results/cnn_metrics.json"
    baseline_path = "ml/results/baseline_metrics.json"
    cnn      = None
    baseline = None
    if os.path.exists(cnn_path):
        with open(cnn_path) as f:
            cnn = json.load(f)
    if os.path.exists(baseline_path):
        with open(baseline_path) as f:
            baseline = json.load(f)
    if cnn is None and baseline is None:
        raise HTTPException(
            status_code=404,
            detail="No metrics found. Run ml/evaluate.py and ml/baseline.py first."
        )
    return {"cnn": cnn, "baselines": baseline}
