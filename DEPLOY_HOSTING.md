# Host the FastAPI API (HTTPS) for Vercel

Your Vercel frontend needs **`VITE_API_BASE_URL`** = `https://<this-service>/api/v1` (no trailing slash).

## $0 (no paid hosting)

**Render free tier** is tight (~512 MB RAM). This repo’s **`render.yaml`** uses **`Dockerfile.free`**, which **drops TensorFlow** so the stack is more likely to fit. You still get **FCN-Wang inference** and **PyTorch Grad-CAM**; only the **MIT-BIH TensorFlow beat Grad-CAM** path is unavailable.

If the free instance still **crashes or fails `/health`**, you do not have to pay:

- **Demo for supervisors:** run **`uvicorn main:app --reload`** on your laptop and use **[Cloudflare Quick Tunnel](https://developers.cloudflare.com/cloudflare-one/connections/connect-apps/do-more-with-tunnels/trycloudflare/)** (`cloudflared tunnel --url http://localhost:8000`) to get a temporary **`https://…trycloudflare.com`** URL, set **`VITE_API_BASE_URL`** on Vercel to `https://…/api/v1`, and redeploy — **$0**, but your PC must stay on and the URL is temporary on the free quick tunnel.

**Full stack (TensorFlow + PyTorch)** on Render usually needs a **paid** instance or another host with **≥2 GB RAM**; use **`Dockerfile`** + `requirements-api.txt` and create a **Web Service** manually (not necessarily the free blueprint).

## 1. Model weights (required)

These files are **gitignored** by default. The API needs them under **`models/artifacts/`**:

- `fastai_fcn_wang.pth`
- `standard_scaler.pkl`
- `mlb.pkl`

**Option A — force-add for deploy (simplest):**

```bash
git add -f models/artifacts/fastai_fcn_wang.pth models/artifacts/standard_scaler.pkl models/artifacts/mlb.pkl
git commit -m "Add model artifacts for cloud deploy"
git push
```

**Option B — disk mount:** On Render, add a **persistent disk**, upload the three files (e.g. via Render Shell), then set env **`CARDIACTWIN_MODELS_DIR`** to that directory (e.g. `/var/data/models/artifacts`) and redeploy.

Optional MIT-BIH Grad-CAM: `models/arrhythmia_cnn_best.h5` (if missing, only that explainability path is skipped).

## 2. Deploy on Render (Docker, $0 blueprint)

1. Push this repo to GitHub (including `Dockerfile.free`, `requirements-api-free.txt`, `render.yaml`).
2. [Render](https://dashboard.render.com) → **New +** → **Blueprint** → connect the repo → apply **`render.yaml`** (uses **free** plan + **`Dockerfile.free`**).
3. Wait for build; open **`https://<name>.onrender.com/health`**. If the service **keeps restarting**, the free tier may still be too small — use the **Cloudflare Tunnel** laptop option above, or save credits from **GitHub Student Pack** / **Render free credits** if you get them later for a paid instance.

**Demo buttons** (`/api/v1/demo/...`) need PTB-XL files under `data/raw/ptbxl/...` on the server. If you did not ship that data, demos will fail until you add those files or only use **CSV / WFDB upload**.

## 3. Point Vercel at the API

1. Vercel → your frontend project → **Settings → Environment Variables**
2. **`VITE_API_BASE_URL`** = `https://<your-render-service>.onrender.com/api/v1`
3. **Redeploy** the frontend (Production) so Vite embeds the variable.

## 4. CORS

`main.py` already allows `allow_origins=["*"]`, so your `*.vercel.app` origin works. Tighten later to your exact Vercel URL.

## 5. Smoke test

```bash
curl -sS "https://<host>/health"
curl -sS "https://<host>/api/v1/demo/normal"
```

If `demo` returns 404/500, use **upload** with your own `.csv` or `.hea+.dat` instead until PTB-XL is on the server.
