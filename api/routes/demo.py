"""
api/routes/demo.py

GET /api/v1/demo/{scenario}

Returns a session_id pre-loaded with a real PTB-XL ECG record so the
frontend can demonstrate the full analysis pipeline without requiring
the user to upload a file.

Scenarios
---------
normal  — Normal sinus rhythm (PTB-XL record 00001)
afib    — Atrial fibrillation  (PTB-XL record 00351)
bbb     — Bundle branch block  (PTB-XL record 00180)
"""

import shutil
import uuid
from pathlib import Path

from fastapi import APIRouter, HTTPException

from config import settings

router = APIRouter(prefix="/api/v1/demo", tags=["Demo"])

_PTBXL_ROOT = Path(
    "data/raw/ptbxl/physionet.org/files/ptb-xl/1.0.3/records100/00000"
)

_SCENARIOS: dict[str, str] = {
    "normal": "00001_lr",   # NORM — normal sinus rhythm
    "afib":   "00351_lr",   # AFIB — atrial fibrillation
    "bbb":    "00180_lr",   # CLBBB — complete left bundle branch block
}


@router.get("/{scenario}")
def load_demo(scenario: str):
    """
    Copy a pre-selected PTB-XL record into session storage and return
    the session_id.  The client then calls POST /analyze/{session_id}
    exactly as it would for a user-uploaded file.
    """
    record_name = _SCENARIOS.get(scenario.lower())
    if record_name is None:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown scenario '{scenario}'. "
                   f"Valid values: {list(_SCENARIOS)}",
        )

    hea = _PTBXL_ROOT / f"{record_name}.hea"
    dat = _PTBXL_ROOT / f"{record_name}.dat"

    if not hea.exists() or not dat.exists():
        raise HTTPException(
            status_code=503,
            detail="Demo records not available on this server. "
                   "Ensure the PTB-XL dataset is present in data/raw/ptbxl/.",
        )

    session_id  = str(uuid.uuid4())
    session_dir = settings.UPLOAD_DIR / session_id
    session_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy(hea, session_dir / f"{record_name}.hea")
    shutil.copy(dat, session_dir / f"{record_name}.dat")

    return {
        "session_id": session_id,
        "scenario":   scenario,
        "record":     record_name,
        "message":    "Demo record loaded. Call POST /analyze/{session_id} to run analysis.",
    }
