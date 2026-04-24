# CardiacTwin FastAPI — CPU inference (Render / Fly / Railway)
FROM python:3.11-bookworm

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Smaller CPU-only PyTorch wheel (install before other deps)
RUN pip install --no-cache-dir \
    torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

COPY . .

RUN mkdir -p storage/uploads storage/sessions models/artifacts

ENV PYTHONUNBUFFERED=1
EXPOSE 8000

# Render and others set PORT; default 8000 for local docker run
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]
