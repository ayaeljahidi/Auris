


## Installation

### 1 — Clone / download the project

```bash
cd backend
```

### 2 — Create a virtual environment (recommended)

```bash
python -m venv .venv
source .venv/bin/activate      # Linux / macOS
.venv\Scripts\activate         # Windows
```

### 3 — Install Python dependencies

```bash
pip install -r requirements.txt
```
``

### 4 — Download models

```bash
python scripts/setup_models.py
```

## Running the Server

```bash
uvicorn backend.main:app --reload --port 8000
```

# Auris — Speech Transcription Service

CPU-optimized audio transcription API built with FastAPI, faster-whisper, and Flan-T5. No system FFmpeg required — audio extraction is handled entirely in-process via PyAV.

---

## Features

- **File upload transcription** — upload any video or audio file, get back a full transcript
- **Live WebSocket transcription** — stream raw PCM audio chunks in real time
- **Flan-T5 correction layer** — optional grammar and spelling correction applied on top of Whisper output
- **Batched segment correction** — all low-confidence Whisper segments corrected in a single Flan-T5 forward pass
- **Critique-gated correction** — high-confidence segments skip Flan-T5 entirely to save CPU cycles
- **FFmpeg-free** — PyAV bundles its own codecs; no system dependencies beyond Python packages
- **Shared thread pool** — one process-wide `ThreadPoolExecutor` reused across all requests

---

## Architecture

```
Upload / WebSocket
       │
       ▼
  PyAV extraction          ← converts any container to 16 kHz mono WAV in-memory
       │
       ▼
  faster-whisper           ← int8 CPU inference, returns segments with confidence metrics
       │
       ▼
  Critique filter          ← skips Flan-T5 for high-confidence segments
       │
       ▼
  Flan-T5 (batched)        ← single generate() call for all low-confidence segments
       │
       ▼
  JSON response
```

### Module overview

| File | Responsibility |
|---|---|
| `main.py` | FastAPI app, routes, WebSocket handler, startup warmup |
| `audio.py` | PyAV extraction, WAV helpers, PCM utilities |
| `transcribe.py` | Whisper inference, Flan-T5 batched correction, critique logic |
| `models.py` | Lazy-loaded model singletons, shared `ThreadPoolExecutor` |
| `config.py` | All tuneable settings via environment variables |

---

## Requirements

```
python >= 3.10
faster-whisper
transformers
torch
av          # PyAV — bundles its own FFmpeg codecs
fastapi
uvicorn
numpy
```

Install:

```bash
pip install faster-whisper transformers torch av fastapi uvicorn numpy
```

---

## Running

```bash
uvicorn auris.main:app --host 0.0.0.0 --port 8000
```

The server pre-loads both models on startup so the first request is fast.

---

## Configuration

All settings are controlled via environment variables. None are required — defaults work out of the box on a CPU machine.

### Model settings

| Variable | Default | Description |
|---|---|---|
| `WHISPER_MODEL` | `base.en` | faster-whisper model size (`tiny.en`, `base.en`, `small.en`, `medium.en`) |
| `FLAN_MODEL` | `google/flan-t5-base` | HuggingFace model ID for the correction layer |
| `FLAN_CACHE_DIR` | `models/flan-t5-base` | Local cache directory for Flan-T5 weights |

### Whisper performance (CPU)

| Variable | Default | Description |
|---|---|---|
| `WHISPER_BEAM_SIZE` | `1` | Beam search width — `1` is greedy and fastest |
| `WHISPER_LANGUAGE` | `en` | Audio language |
| `WHISPER_NUM_WORKERS` | `1` | Parallel Whisper workers — `1` is optimal on CPU |
| `WHISPER_CPU_THREADS` | `4` | CPU threads for Whisper — set to your core count |

### Flan-T5 correction

| Variable | Default | Description |
|---|---|---|
| `FLAN_ENABLED` | `true` | Enable correction on file uploads |
| `FLAN_ENABLED_LIVE` | `false` | Enable correction on live WebSocket (slow on CPU; off by default) |
| `FLAN_MAX_TOKENS` | `64` | Max output tokens per sentence |
| `FLAN_NUM_BEAMS` | `1` | Beam width for Flan-T5 — `1` is greedy/fastest |

### Critique thresholds

Segments that pass all three checks are kept without correction, saving CPU.

| Variable | Default | Description |
|---|---|---|
| `CRITIQUE_NO_SPEECH_THRESHOLD` | `0.5` | Skip correction if `no_speech_prob` exceeds this |
| `CRITIQUE_AVG_LOGPROB_THRESHOLD` | `-0.5` | Correct if `avg_logprob` falls below this |
| `CRITIQUE_COMPRESSION_RATIO_MAX` | `2.4` | Correct if `compression_ratio` exceeds this (hallucination signal) |

### WebSocket

| Variable | Default | Description |
|---|---|---|
| `WS_LIVE_TIMEOUT` | `300` | Seconds of idle silence before the WebSocket closes |

---

## API

### `GET /health`

Returns model load status and configuration.

```json
{
  "status": "ok",
  "version": "10.0",
  "whisper_model": "base.en",
  "flan_enabled": true,
  "flan_enabled_live": false,
  "flan_model": "google/flan-t5-base",
  "device": "cpu",
  "executor_workers": 4
}
```

### `POST /transcribe`

Upload any audio or video file. Returns the Whisper transcript and the Flan-T5 corrected version with per-segment timing.

**Request:** `multipart/form-data` with a `file` field.

```bash
curl -X POST http://localhost:8000/transcribe \
  -F "file=@recording.mp4"
```

**Response:**

```json
{
  "status": "ok",
  "filename": "recording.mp4",
  "duration_sec": 42.5,
  "whisper": {
    "text": "...",
    "word_count": 120,
    "segments": [
      { "start": 0.0, "end": 3.2, "text": "Hello world" }
    ]
  },
  "correction": {
    "corrected": "...",
    "enabled": true,
    "model": "google/flan-t5-base",
    "latency_ms": 310,
    "critique_stats": {
      "corrected": 3,
      "kept": 12,
      "total": 15
    }
  },
  "timing": {
    "extract_ms": 45,
    "whisper_ms": 1820,
    "total_ms": 1865
  }
}
```

### `WebSocket /ws/live`

Stream raw 16-bit mono 16 kHz PCM chunks. Send `b"__END__"` when done recording.

**Client → server:**
- Binary frames of raw PCM (any chunk size)
- `b"__END__"` to signal end of recording

**Server → client:**
```json
{ "type": "partial", "text": "Recording… speak now" }
{ "type": "status",  "msg": "Running Whisper…" }
{ "type": "final",   "whisper": { ... }, "correction": { ... } }
{ "type": "error",   "msg": "Recording timeout" }
```

---

## Performance notes

**Choosing a Whisper model size on CPU:**

| Model | ~VRAM | Relative speed | Use case |
|---|---|---|---|
| `tiny.en` | 75 MB | fastest | Real-time, short clips |
| `base.en` | 145 MB | fast | Default — good balance |
| `small.en` | 465 MB | medium | Better accuracy |
| `medium.en` | 1.5 GB | slow | Highest accuracy |

**Disabling Flan-T5** with `FLAN_ENABLED=false` cuts transcription latency roughly in half on CPU. Whisper `base.en` accuracy is already good for clear speech.

**Critique thresholds** control how aggressively Flan-T5 is applied. Lowering `CRITIQUE_AVG_LOGPROB_THRESHOLD` (e.g. to `-0.8`) means fewer segments get corrected and the response is faster. Raising `CRITIQUE_COMPRESSION_RATIO_MAX` (e.g. to `3.0`) reduces hallucination correction.

---

## Frontend

If a `frontend/` directory exists at the project root, it is served as a static site on `/`. The API lives on `/transcribe`, `/ws/live`, and `/health`.



