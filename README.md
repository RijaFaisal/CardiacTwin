# ECG ML Project

This project provides an ECG (Electrocardiogram) analysis and prediction system with a machine learning backend and a modern web frontend.

## Features

- ECG signal analysis and delineation
- Machine learning-based prediction using FCN models
- Web interface for uploading and analyzing ECG data
- REST API for programmatic access
- Support for various ECG formats (WFDB, CSV)

## Prerequisites

- Python 3.8+
- Node.js 16+
- Git

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/umarhussain47/ecg-ml.git
   cd ecg-ml
   ```

### Backend Setup

1. Navigate to the backend directory:
   ```bash
   cd ecg-back
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   # On Windows:
   .venv\Scripts\activate
   # On macOS/Linux:
   source .venv/bin/activate
   ```

3. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up environment variables (optional):
   Create a `.env` file in the `ecg-back` directory with any necessary configuration.

5. Download or place model artifacts:
   - Place your trained model files (`.pth`) in `models/artifacts/`
   - If models are not included, follow the training instructions in the documentation

### Frontend Setup

1. Navigate to the frontend directory:
   ```bash
   cd ecg-front
   ```

2. Install Node.js dependencies:
   ```bash
   npm install
   ```

3. Start the development server:
   ```bash
   npm run dev
   ```

## Running the Project

1. Start the backend server:
   ```bash
   cd ecg-back
   python main.py
   ```
   The backend will run on `http://localhost:5000` (or configured port).

2. Start the frontend (in a separate terminal):
   ```bash
   cd ecg-front
   npm run dev
   ```
   The frontend will run on `http://localhost:5173` (or configured port).

3. Open your browser and navigate to the frontend URL to use the application.

## API Documentation

The backend provides REST API endpoints for:
- `/api/upload` - Upload ECG files
- `/api/analyze` - Analyze ECG data
- `/api/predict` - Run ML predictions
- `/api/export` - Export results

## Project Structure

```
ecg-back/          # Python backend
├── api/           # API routes
├── core/          # Core functionality
│   ├── analysis/  # ECG analysis modules
│   ├── inference/ # ML inference
│   └── loader/    # Data loading
├── models/        # ML models and artifacts
├── storage/       # File storage
└── schemas/       # Data schemas

ecg-front/         # React frontend
├── src/
│   ├── components/
│   ├── api/
│   └── types/
└── public/
```

## Development

### Backend Development
- Use `python main.py` to run the development server
- Add new routes in `api/routes/`
- Implement analysis logic in `core/analysis/`

### Frontend Development
- Use `npm run dev` for hot-reloading
- Components are in `src/components/`
- API calls are in `src/api/`

## Contributing
