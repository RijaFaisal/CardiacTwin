import os
from fastapi import APIRouter, HTTPException
from config import settings
from core.inference.fcn_wang_runner import ml_pipeline
from core.loader.wfdb_parser import load_native
import scipy.signal

router = APIRouter()

@router.post("/predict/{session_id}")
async def run_inference(session_id: str):
    try:
        session_dir = settings.UPLOAD_DIR / session_id
        
        if not os.path.exists(session_dir):
            raise HTTPException(status_code=404, detail="Session folder not found.")

        # Find the .hea file in the folder
        hea_file = next((f for f in os.listdir(session_dir) if f.endswith('.hea')), None)
        
        if not hea_file:
            raise HTTPException(status_code=400, detail="No .hea file found in session folder.")

        # WFDB needs the path to the .hea file WITHOUT the extension
        base_path = str(session_dir / hea_file[:-4])

        # Parse and Predict
        signal_raw, native_sr = load_native(base_path)
        signal_ai_input = scipy.signal.resample(signal_raw, 1000)
        prediction = ml_pipeline.predict(signal_ai_input)

        return {
            "status": "success", 
            "session_id": session_id,
            "data": prediction
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))