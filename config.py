import os
from pathlib import Path

class Settings:
    PROJECT_NAME: str = "ECG Wang Backend"
    
    # Base directories
    BASE_DIR = Path(__file__).resolve().parent
    # Cloud: set CARDIACTWIN_MODELS_DIR to a persistent disk path containing the .pth / .pkl files
    MODELS_DIR = (
        Path(os.environ["CARDIACTWIN_MODELS_DIR"]).resolve()
        if os.environ.get("CARDIACTWIN_MODELS_DIR")
        else (BASE_DIR / "models" / "artifacts")
    )
    UPLOAD_DIR = BASE_DIR / "storage" / "uploads"
    
    # Artifact paths
    MODEL_WEIGHTS_PATH = MODELS_DIR / "fastai_fcn_wang.pth"
    SCALER_PATH = MODELS_DIR / "standard_scaler.pkl"
    MLB_PATH = MODELS_DIR / "mlb.pkl"

    def __init__(self):
        # Ensure upload directory exists
        os.makedirs(self.UPLOAD_DIR, exist_ok=True)

settings = Settings()