


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



Whisper transcribes the audio and returns segments. Each segment includes confidence metrics: no_speech_prob, avg_logprob, and compression_ratio.
The critique filter checks those metrics on each segment. If a segment looks confident enough, it gets passed through untouched ("kept"). If it looks low-quality — low log probability, high compression ratio — it gets flagged for correction.
Flan-T5 corrects only the flagged segments. All of them are batched into a single generate() call (not one call per segment), so the model overhead is paid once regardless of how many segments need fixing.
The corrected segments replace the originals, and the full text is reassembled and returned alongside the raw Whisper output.

