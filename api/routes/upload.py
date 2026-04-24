import shutil
import uuid
import os
from fastapi import APIRouter, UploadFile, File, HTTPException
from config import settings

router = APIRouter()


# ── Existing endpoint — UNCHANGED ────────────────────────────────────────────
@router.post("/upload")
async def upload_wfdb(
    header_file: UploadFile = File(..., description="The .hea file"),
    data_file: UploadFile = File(..., description="The .dat file")
):
    if not header_file.filename.endswith('.hea') or not data_file.filename.endswith('.dat'):
        raise HTTPException(status_code=400, detail="Invalid file extensions. Need .hea and .dat")

    session_id = str(uuid.uuid4())

    # Create a sub-folder for this specific session
    session_dir = settings.UPLOAD_DIR / session_id
    os.makedirs(session_dir, exist_ok=True)

    # Save files with their ORIGINAL names inside that folder
    for file in [header_file, data_file]:
        save_path = session_dir / file.filename
        with open(save_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

    return {
        "status": "success",
        "session_id": session_id,
        "message": "WFDB pair uploaded successfully."
    }


# ── NEW: CSV upload endpoint ──────────────────────────────────────────────────
@router.post("/upload/csv")
async def upload_csv(
    file: UploadFile = File(
        ...,
        description=(
            "A single ECG CSV file. Accepted sources: GE MUSE XML export, "
            "PhysioNet large-scale 12-lead DB (Chapman-Shaoxing / CPSC), "
            "Philips PageWriter, Schiller CARDIOVIT. "
            "Must contain 12 ECG lead columns (missing leads are zero-padded automatically)."
        ),
    )
):
    if not (file.filename or "").lower().endswith(".csv"):
        raise HTTPException(
            status_code=400,
            detail=f"File '{file.filename}' is not a .csv. Export your ECG device data as CSV first."
        )

    session_id = str(uuid.uuid4())
    session_dir = settings.UPLOAD_DIR / session_id
    os.makedirs(session_dir, exist_ok=True)

    save_path = session_dir / file.filename
    with open(save_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    return {
        "status": "success",
        "session_id": session_id,
        "message": "CSV uploaded successfully."
    }