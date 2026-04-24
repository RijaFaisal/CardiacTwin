# main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Existing routes
from api.routes import upload, predict

# New routes
from api.routes import analyze, export

app = FastAPI(title="ECG Analysis API", version="1.0.0")

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update to ["http://localhost:5173"] in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Existing routers
app.include_router(upload.router, prefix="/api/v1", tags=["File Management"])
app.include_router(predict.router, prefix="/api/v1", tags=["ML Inference"])

# New routers
app.include_router(analyze.router, prefix="/api/v1", tags=["Analysis"])
app.include_router(export.router, prefix="/api/v1", tags=["Export"])

@app.get("/")
def read_root():
    return {"message": "ECG Backend is running. Inference engine loaded."}