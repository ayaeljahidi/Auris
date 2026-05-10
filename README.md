.venv\Scripts\activate  
pip install modelscope[audio] -f https://modelscope.oss-cn-beijing.aliyuncs.com/releases/repo.html
python scripts/setup_models.py 
uvicorn backend.main:app --reload --port 8000 --reload-dir backend

# Auris — Speech Analysis App

## Project Structure

```
project/
├── backend/          ← FastAPI Python package
│   ├── __init__.py
│   ├── main.py       ← FastAPI app (fixed WS endpoint)
│   ├── audio.py      ← PyAV audio extraction
│   ├── config.py     ← Environment config
│   ├── emotion.py    ← Wav2Vec2 emotion detection
│   ├── models.py     ← Model singletons
│   ├── transcribe.py ← Whisper + Flan-T5 pipeline
│   └── qwen_questions.py ← Qwen QG via Ollama
├── frontend/         ← React + Tailwind (Vite)
│   ├── src/
│   │   ├── sections/
│   │   │   ├── UploadPage.tsx   ← fixed: uses /transcribe (relative)
│   │   │   └── LivePage.tsx     ← fixed: uses /ws/live (relative, protocol-aware)
│   │   └── ...
│   └── vite.config.ts           ← fixed: proxy to backend
└── requirements.txt
```

## What Was Fixed

### 1. `backend/main.py` — WebSocket live endpoint (critical bug)
The browser's `MediaRecorder` sends **audio/webm** (Opus-encoded), NOT raw int16 PCM.
The original code did `np.frombuffer(data, dtype=np.int16)` on the compressed webm
bytes, which produced garbage audio and silent/empty Whisper results.

**Fix:** Accumulate all webm chunks into a buffer, then on `__END__` decode the
reassembled webm container with `extract_audio_to_numpy()` via PyAV — exactly the
same code path used by the `/transcribe` endpoint.

### 2. `backend/__init__.py` — missing package marker
The backend uses relative imports (`from .audio import …`) so it must be a Python
package. Without `__init__.py` you get `ImportError: attempted relative import with
no known parent package`.

**Fix:** Added `backend/__init__.py`.

### 3. `frontend/vite.config.ts` — no dev proxy
The frontend had `fetch('http://localhost:8000/transcribe')` hardcoded. This fails
whenever the ports differ, in production builds, or behind a reverse proxy.

**Fix:** Added a Vite `server.proxy` config that forwards `/transcribe`, `/emotion`,
`/health`, and `/ws/live` to `localhost:8000`. The frontend code now uses relative
paths (`/transcribe`, `/ws/live`).

### 4. `frontend/src/sections/UploadPage.tsx` — hardcoded URL
Changed `http://localhost:8000/transcribe` → `/transcribe`.

### 5. `frontend/src/sections/LivePage.tsx` — hardcoded WS URL
Changed `ws://localhost:8000/ws/live` → dynamic `${wsProtocol}//${window.location.host}/ws/live`
so it works over HTTPS/WSS in production too.

## Setup

### Backend

```bash
# Create virtualenv (recommended)
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Optional: install Ollama for question generation
# https://ollama.com — then: ollama pull qwen2.5:1.5b

# Start the backend (from project root, so `backend` is the package)
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

### Frontend

```bash
cd frontend
npm install
npm run dev        # dev server at http://localhost:3000
# or
npm run build      # production build → frontend/dist/
```

When running `npm run dev`, Vite proxies all `/transcribe`, `/emotion`, `/health`,
and `/ws/live` requests to the FastAPI server at `localhost:8000`.

For production, copy `frontend/dist/` next to the backend — FastAPI serves it via
`StaticFiles` from `frontend/dist/`.

## Environment Variables (optional)

| Variable | Default | Description |
|---|---|---|
| `WHISPER_MODEL` | `base.en` | Whisper model size |
| `FLAN_ENABLED` | `true` | Enable Flan-T5 correction |
| `FLAN_ENABLED_LIVE` | `false` | Enable Flan-T5 in live mode |
| `EMOTION_ENABLED` | `true` | Enable emotion detection |
| `QG_ENABLED_LIVE` | `true` | Enable question generation in live mode |
