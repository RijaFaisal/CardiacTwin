# Cardiac Digital Twin — ECG Analysis System

Final Year Project: AI-powered ECG analysis with a cardiac digital twin simulator.

## Features

- **12-lead ECG analysis** — NeuroKit2 delineation (PR, QRS, QTc intervals, HRV)
- **CNN classifier** — FCN-Wang architecture trained on MIT-BIH Arrhythmia Database with inter-patient split
- **Grad-CAM explainability** — beat-level saliency heatmap overlaid on the waveform
- **3D Cardiac Digital Twin** — Three.js heart model with patient-specific BPM animation
- **Digital Twin Simulator** — pathology/treatment before-vs-after cardiac parameter comparison
- **Model performance panel** — CNN vs Transformer vs SVM vs Logistic Regression macro-F1 comparison
- **PDF clinical report** — exportable ReportLab PDF with verdict, metrics, and AI predictions
- **Multi-format upload** — WFDB (`.hea`/`.dat`) and CSV

## Prerequisites

- Python 3.10+
- Node.js 18+

## Quick Start

### 1. Backend

```bash
# From repo root
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt

uvicorn main:app --reload --port 8000
```

API runs at `http://localhost:8000`. Interactive docs at `/docs`.

### 2. Frontend

```bash
cd ecg-front
npm install
npm run dev
```

UI runs at `http://localhost:5173`.

## ML Training (optional — pretrained model included)

```bash
source venv/bin/activate

# Build inter-patient dataset split
python ml/build_dataset.py

# Train CNN (FCN-Wang)
python ml/train.py

# Train Transformer encoder
python ml/train_transformer.py

# Evaluate CNN + bootstrap CI + McNemar test
python ml/evaluate.py

# SVM + Logistic Regression baselines
python ml/baseline.py
```

Results are written to `ml/results/` and served via `GET /api/v1/simulate/metrics`.

## Project Structure

```
.
├── main.py                  # FastAPI app entry point
├── api/routes/              # upload, analyze, export, simulate
├── core/
│   ├── analysis/            # NeuroKit2 delineation, condition mapper
│   ├── inference/           # FCN-Wang runner, Grad-CAM
│   └── loader/              # WFDB + CSV auto-detect loader
├── ml/                      # Training, evaluation, augmentation scripts
│   ├── results/             # cnn_metrics.json, baseline_metrics.json
│   └── data/                # X_train/val/test .npy (gitignored)
├── models/                  # Trained model weights
├── schemas/                 # Pydantic schemas (AnalysisResponse etc.)
├── storage/
│   ├── sessions/            # Persisted analysis JSON (for PDF export)
│   └── uploads/             # Uploaded ECG files (gitignored)
└── ecg-front/               # React + TypeScript + Vite frontend
    └── src/
        ├── components/      # Dashboard, AboveFold, HeartVisualization, SimulatePanel, ThirdScroll
        ├── api/             # ecgClient.ts
        └── types/           # analysis.ts
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/v1/upload` | Upload WFDB pair |
| POST | `/api/v1/upload/csv` | Upload CSV |
| POST | `/api/v1/analyze/{session_id}` | Full ECG analysis |
| POST | `/api/v1/analyze/{session_id}?explain=true` | Analysis + Grad-CAM |
| GET  | `/api/v1/analyze/{session_id}/signal-quality` | Signal quality only |
| POST | `/api/v1/export/{session_id}/pdf` | PDF clinical report |
| POST | `/api/v1/simulate/pathology` | Simulate pathology parameters |
| POST | `/api/v1/simulate/treatment` | Simulate treatment response |
| GET  | `/api/v1/simulate/metrics` | CNN vs baseline model metrics |
| GET  | `/health` | Health check + model load status |

## Disclaimer

This system is for research and educational purposes only. It does not constitute a medical diagnosis. All findings must be reviewed by a qualified healthcare professional.
