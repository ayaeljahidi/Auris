# Auris

> **Speech Intelligence Pipeline**: Upload a video/audio file or record live, and get a dual transcription (Vosk + faster-whisper) with voice-activity detection powered by MarbleNet, followed by a Flan-T5 grammar correction layer.

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Project Structure](#project-structure)
3. [How the Pipeline Works](#how-the-pipeline-works)
4. [Prerequisites](#prerequisites)
5. [Installation](#installation)
6. [Running the Server](#running-the-server)
7. [Testing](#testing)
8. [Using the Frontend](#using-the-frontend)
9. [API Reference](#api-reference)
10. [Configuration (Environment Variables)](#configuration-environment-variables)
11. [Frontend Structure](#frontend-structure)
12. [Model Notes](#model-notes)

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                         Vosper v8                               │
│                                                                 │
│  Frontend (HTML + Tailwind CSS + Vanilla JS)                    │
│    ├── Upload tab   → POST /transcribe                          │
│    └── Live tab     → WebSocket /ws/live                        │
│                              │                                  │
│  Backend (FastAPI / Python)  │                                  │
│    ├── main.py     — routes, WebSocket handler, startup         │
│    ├── config.py   — all env-configurable settings              │
│    ├── models.py   — lazy singleton model loaders               │
│    ├── audio.py    — FFmpeg extraction, WAV/PCM helpers         │
│    ├── vad.py      — MarbleNet VAD (batched ONNX)               │
│    └── transcribe.py — Vosk + faster-whisper engines            │
│                        + Flan-T5 correction layer               │
└─────────────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
vosper/
│
├── backend/                      ← FastAPI application (Python package)
│   ├── main.py                   ← App entry point, all HTTP/WS routes
│   ├── config.py                 ← Settings read from environment variables
│   ├── models.py                 ← Lazy singleton loaders (Vosk, Whisper, MarbleNet, Flan-T5)
│   ├── audio.py                  ← FFmpeg audio extraction + WAV/PCM helpers
│   ├── vad.py                    ← MarbleNet VAD — batched ONNX inference
│   └── transcribe.py             ← Vosk + faster-whisper transcription engines + Flan-T5 correction
│
├── scripts/
│   └── setup_models.py           ← Downloads & verifies all required models
│
├── models/                       ← Downloaded model files (created by setup)
│   ├── vosk/small/               ← Vosk small English model
│   └── marblenet/                ← MarbleNet VAD .nemo file
│
├── requirements.txt
└── README.md
```

---

## How the Pipeline Works

### Upload mode

```
User uploads file
      │
      ▼
  [1] FFmpeg
      Converts any video/audio format to 16-kHz mono PCM WAV.
      Runs as a subprocess with a 120-second timeout.
      │
      ▼
  [2] MarbleNet VAD  (batched ONNX inference)
      Segments the audio into speech / silence regions.
      Builds a batch of 20 ms frames, runs a single ONNX call,
      then merges close segments and adds 100 ms padding.
      Produces a trimmed WAV containing only speech.
      │
      ├──────────────────────┐
      ▼                      ▼
  [3a] Vosk             [3b] faster-whisper     ← PARALLEL (asyncio.gather)
       KaldiRecognizer        WhisperModel
       streams chunks,        transcribes the
       returns word-level      full trimmed WAV
       confidence scores       with timestamps
      │                      │
      └──────────┬───────────┘
                 ▼
  [4] Flan-T5 correction  (google/flan-t5-base by default)
      Takes the Whisper output text, splits it into sentences,
      corrects grammar and spelling on each sentence,
      then reassembles into a clean paragraph.
      Can be disabled with FLAN_ENABLED=false.
                 │
                 ▼
           JSON response
           (vad_segments, vosk, whisper, correction, timing)
```

Total latency = t(FFmpeg) + t(VAD) + **max**(t(Vosk), t(Whisper)) + t(Flan-T5)

### Live mode

```
Browser mic → AudioWorklet (Float32 → Int16 PCM)
      │
      ▼  (WebSocket frames)
  Vosk KaldiRecognizer   ← streams partial results back in real-time
      │
  User clicks "Stop & transcribe"
      │   b"__END__" sentinel sent
      ▼
  MarbleNet VAD + faster-whisper  ← run on accumulated buffer
      │
  Flan-T5 correction              ← run on Whisper final text
      │
  Final result sent back over WebSocket
```

---

## Prerequisites

| Requirement | Version | Notes |
|-------------|---------|-------|
| Python      | 3.9+    | Required |
| FFmpeg      | Any     | Must be on system `PATH` |
| pip         | Any     | For Python packages |

**FFmpeg install:**
```bash
# Ubuntu / Debian
sudo apt install ffmpeg

# macOS
brew install ffmpeg

# Windows
choco install ffmpeg    # or download from https://ffmpeg.org/download.html
```

---

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

This script will:
1. ✓ Check FFmpeg is available
2. ✓ Verify all Python packages are installed
3. ✓ Download the **Vosk** small English model (~50 MB) to `models/vosk/small/`
4. ✓ Download **MarbleNet VAD** from NVIDIA NGC (~7 MB) to `models/marblenet/`
5. ✓ Run a warmup pass of **faster-whisper** (downloads `base.en` model ~150 MB automatically)

---

## Running the Server

```bash
uvicorn backend.main:app --reload --port 8000
```

Then open your browser at **http://localhost:8000**

The server serves the frontend automatically from the `frontend/` directory.

**Production mode (no reload, multiple workers):**
```bash
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --workers 2
```

---

## Testing

### Standard pipeline test (no correction)

Run the server with Flan-T5 disabled, then call the `/transcribe` endpoint directly:

```cmd
# Windows
set FLAN_ENABLED=false && uvicorn backend.main:app --reload --port 8000
```

```bash
# Linux / macOS
FLAN_ENABLED=false uvicorn backend.main:app --reload --port 8000
```

Then test with curl (replace `your_audio.mp4` with your file):

```bash
curl -X POST http://localhost:8000/transcribe \
  -F "file=@your_audio.mp4"
```

The response will contain `vosk` and `whisper` results **without** the `correction` field active:

```json
{
  "status": "ok",
  "whisper": { "text": "the weather is vary good today..." },
  "vosk":    { "text": "the weather is vary good today..." },
  "correction": { "corrected": "the weather is vary good today...", "enabled": false }
}
```

---

### Pipeline test with Flan-T5 correction (default)

Run the server normally — Flan-T5 is **enabled by default**:

```cmd
uvicorn backend.main:app --reload --port 8000
```

Then send the same request:

```bash
curl -X POST http://localhost:8000/transcribe \
  -F "file=@your_audio.mp4"
```

The response now includes a `correction` block with the cleaned-up text and its timing:

```json
{
  "status": "ok",
  "whisper":  { "text": "the weather is vary good today..." },
  "vosk":     { "text": "the weather is vary good today..." },
  "correction": {
    "corrected":  "The weather is very good today.",
    "enabled":    true,
    "model":      "google/flan-t5-base",
    "latency_ms": 4200
  },
  "timing": {
    "ffmpeg_ms":     340,
    "vad_ms":        120,
    "parallel_ms":   1850,
    "correction_ms": 4200,
    "total_ms":      6510
  }
}
```

---

### Flan-T5 standalone test (compare base vs large)

Before integrating, you can benchmark both models on a sample transcription:

```cmd
pip install transformers torch sentencepiece accelerate
python test_flan_t5.py
```

This runs both `flan-t5-base` and `flan-t5-large` across 3 prompt styles on 10 sample sentences and prints a full comparison table of quality and latency. Use the results to decide which model to set via `FLAN_MODEL`.

---

### Switch to flan-t5-large for higher quality

```cmd
# Windows
set FLAN_MODEL=google/flan-t5-large && uvicorn backend.main:app --port 8000
```

```bash
# Linux / macOS
FLAN_MODEL=google/flan-t5-large uvicorn backend.main:app --port 8000
```

> ⚠️ Large is ~800 MB and 3–4x slower than base on CPU. Only recommended if you have a GPU.

---

## Using the Frontend

### Upload tab

1. Drag & drop a video or audio file into the drop zone, or click **Choose file**
2. Preview your file in the built-in video player
3. Click **Run transcription**
4. Watch the pipeline indicators (FFmpeg → VAD → Vosk ‖ Whisper)
5. Results appear in the dashboard below:
   - **Stats row** — duration, VAD segment count, word counts
   - **Whisper panel** — full transcript + timed segments
   - **Vosk panel** — full transcript + word-confidence chips
   - **VAD timeline** — visual representation of speech segments
   - **Pipeline timing** — bar chart of each stage's duration
6. Use **Export JSON** to download the full result object
7. Use **New session** to start over

### Live tab

1. Click **Start recording** — browser will ask for microphone permission
2. Speak — Vosk partial transcripts appear in real-time
3. Click **Stop & transcribe** when done
4. MarbleNet VAD + Whisper run on the accumulated audio
5. Final results appear in the dashboard

---

## API Reference

### `GET /health`

Returns system and model status.

```json
{
  "status": "ok",
  "version": "8.0",
  "vosk_ready": true,
  "whisper_model": "base.en",
  "marblenet_ready": true
}
```

---

### `POST /transcribe`

Upload a video or audio file. Runs the full pipeline.

**Request:** `multipart/form-data` with field `file`

**Response:**
```json
{
  "status": "ok",
  "filename": "recording.mp4",
  "duration_sec": 32.5,
  "vad_segments": [
    { "start": 0.24, "end": 4.82, "confidence": 0.91 },
    { "start": 6.10, "end": 12.40, "confidence": 0.88 }
  ],
  "whisper": {
    "text": "Hello world this is a test…",
    "word_count": 7,
    "segments": [
      { "start": 0.24, "end": 2.10, "text": "Hello world" }
    ]
  },
  "vosk": {
    "text": "hello world this is a test",
    "word_count": 6,
    "words": [
      { "word": "hello", "start": 0.27, "conf": 0.98 }
    ]
  },
  "correction": {
    "corrected":  "Hello world, this is a test.",
    "enabled":    true,
    "model":      "google/flan-t5-base",
    "latency_ms": 3800
  },
  "timing": {
    "ffmpeg_ms":     340,
    "vad_ms":        120,
    "parallel_ms":   1850,
    "correction_ms": 3800,
    "total_ms":      6110
  }
}
```

---

### `WebSocket /ws/live`

Streaming live transcription.

**Protocol:**
| Direction | Payload | Meaning |
|-----------|---------|---------|
| Client → Server | `ArrayBuffer` (Int16 PCM) | Raw 16-bit mono 16-kHz audio chunks |
| Client → Server | `b"__END__"` | End of recording |
| Server → Client | `{"type":"partial","text":"…"}` | Vosk partial result |
| Server → Client | `{"type":"status","msg":"…"}` | Status update |
| Server → Client | `{"type":"final","whisper":{…},"vad_segments":[…],"vosk_text":"…","correction":{…}}` | Final result with correction |
| Server → Client | `{"type":"error","msg":"…"}` | Error |

---

### `GET /api/docs`

Interactive Swagger UI (FastAPI auto-generated).

---

## Configuration (Environment Variables)

All settings are in `backend/config.py` and can be overridden via environment variables.

| Variable             | Default                              | Description                          |
|----------------------|--------------------------------------|--------------------------------------|
| `VOSK_MODEL_PATH`    | `models/vosk/small`                  | Path to the Vosk model directory     |
| `WHISPER_MODEL`      | `base.en`                            | faster-whisper model identifier      |
| `MARBLENET_PATH`     | `models/marblenet/marblenet-vad.onnx`| Path to the MarbleNet ONNX file      |
| `WHISPER_LANGUAGE`   | `en`                                 | Transcription language               |
| `WHISPER_BEAM_SIZE`  | `1`                                  | Beam size (1 = fastest, lower accuracy) |
| `WHISPER_NUM_WORKERS`| `2`                                  | Parallel workers for Whisper         |
| `WHISPER_CPU_THREADS`| `4`                                  | CPU threads for Whisper              |
| `VAD_THRESHOLD`      | `0.5`                                | Speech probability threshold (0–1)   |
| `VAD_MIN_SPEECH_MS`  | `100`                                | Minimum speech segment in ms         |
| `VAD_MIN_SILENCE_MS` | `200`                                | Minimum silence gap to split in ms   |
| `FFMPEG_TIMEOUT`     | `120`                                | FFmpeg subprocess timeout in seconds |
| `ONNX_THREADS`       | `4`                                  | ONNX Runtime CPU threads             |
| `FLAN_ENABLED`       | `true`                               | Set to `false` to skip correction    |
| `FLAN_MODEL`         | `google/flan-t5-base`                | Flan-T5 model (`flan-t5-base` or `flan-t5-large`) |
| `FLAN_MAX_TOKENS`    | `512`                                | Max tokens generated per sentence    |
| `FLAN_NUM_BEAMS`     | `4`                                  | Beam search width (higher = better quality, slower) |

**Example — use a larger Whisper model:**
```bash
WHISPER_MODEL=medium.en uvicorn backend.main:app --port 8000
```

**Example — disable Flan-T5 correction:**
```cmd
# Windows
set FLAN_ENABLED=false && uvicorn backend.main:app --port 8000
```
```bash
# Linux / macOS
FLAN_ENABLED=false uvicorn backend.main:app --port 8000
```

**Example — switch to flan-t5-large:**
```bash
FLAN_MODEL=google/flan-t5-large uvicorn backend.main:app --port 8000
```

---

## Frontend Structure

```
frontend/
├── index.html        ← Semantic HTML skeleton. No inline styles or scripts.
│                       Loads Tailwind CSS (CDN), style.css, and app.js.
│                       Contains all templates/layouts for Upload, Live, Dashboard.
│
├── css/style.css     ← Custom CSS only.
│                       Handles things Tailwind can't: ::before backgrounds,
│                       custom scrollbars, keyframe animations, complex
│                       pseudo-element effects, pipeline/badge/waveform styles.
│
└── js/app.js         ← All application logic (pure Vanilla JS, no framework).
                        Organized into clear sections:
                          • CONFIG & State — single source of truth
                          • Health check   — /health polling
                          • Tab switching
                          • File upload flow (drop, pick, transcribe, progress)
                          • Live recording  (getUserMedia, AudioWorklet, WebSocket)
                          • Dashboard rendering (stats, transcripts, VAD, timing)
                          • Actions (export, reset, copy)
                          • Utilities
```

---

## Model Notes

| Model | Size | Purpose | Source |
|-------|------|---------|--------|
| Vosk small-en | ~50 MB | Fast real-time speech recognition | [alphacephei.com](https://alphacephei.com/vosk/models) |
| MarbleNet | ~7 MB | Voice activity detection | [NVIDIA NGC](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/nemo/models/vad_multilingual_frame_marblenet) |
| Whisper base.en | ~150 MB | High-accuracy transcription | [OpenAI (via faster-whisper)](https://github.com/SYSTRAN/faster-whisper) |
| Flan-T5 base | ~250 MB | Grammar & spelling correction | [Google (via HuggingFace)](https://huggingface.co/google/flan-t5-base) |
| Flan-T5 large | ~800 MB | Higher quality correction (slower) | [Google (via HuggingFace)](https://huggingface.co/google/flan-t5-large) |

To use a **larger Whisper model** for better accuracy, set `WHISPER_MODEL` to one of:
`tiny`, `base`, `small`, `medium`, `large-v3` (append `.en` for English-only variants, e.g. `small.en`).

To use **Flan-T5 large** for better correction quality, set `FLAN_MODEL=google/flan-t5-large`.
Note that large is ~3–4x slower than base on CPU — a GPU is recommended for production use.

---

## Troubleshooting

**`RuntimeError: Vosk model not found`**
→ Run `python scripts/setup_models.py`

**`MarbleNet not found — using fallback VAD`**
→ The setup script couldn't download MarbleNet. The app still works, using a pass-through (whole file treated as speech). Retry the setup script or download manually from NGC.

**`FFmpeg failed`**
→ Make sure `ffmpeg` is installed and on your `PATH`. Test with `ffmpeg -version`.

**Port 8000 already in use**
→ Change the port: `uvicorn backend.main:app --port 8001`

**Microphone not working in Live mode**
→ Your browser must be served over HTTPS or `localhost` for `getUserMedia` to work. Running locally on `localhost:8000` is fine.

**Flan-T5 slow on first request**
→ The model is downloaded from HuggingFace on first use (~250 MB for base). Subsequent runs use the local cache. To pre-download manually:
```bash
python -c "from transformers import T5Tokenizer, T5ForConditionalGeneration; T5Tokenizer.from_pretrained('google/flan-t5-base'); T5ForConditionalGeneration.from_pretrained('google/flan-t5-base')"
```

**Want to skip correction for faster results**
→ Set `FLAN_ENABLED=false` before starting the server. The `correction` field will still appear in the response but with `"enabled": false` and the original text unchanged.