"""Vosper — Configuration"""
import os

# ── Model paths ────────────────────────────────────────────────────────────────
VOSK_PATH       = os.environ.get("VOSK_MODEL_PATH",  "models/vosk/small")
WHISPER_MODEL   = os.environ.get("WHISPER_MODEL",    "base.en")
MARBLENET_PATH  = os.environ.get("MARBLENET_PATH",   "models/marblenet/marblenet-vad.onnx")

# ── Whisper settings ───────────────────────────────────────────────────────────
WHISPER_BEAM_SIZE   = int(os.environ.get("WHISPER_BEAM_SIZE",   "1"))
WHISPER_LANGUAGE    = os.environ.get("WHISPER_LANGUAGE",        "en")
WHISPER_NUM_WORKERS = int(os.environ.get("WHISPER_NUM_WORKERS", "2"))
WHISPER_CPU_THREADS = int(os.environ.get("WHISPER_CPU_THREADS", "4"))

# ── VAD settings ───────────────────────────────────────────────────────────────
VAD_THRESHOLD       = float(os.environ.get("VAD_THRESHOLD",       "0.5"))
VAD_MIN_SPEECH_MS   = int(os.environ.get("VAD_MIN_SPEECH_MS",     "100"))
VAD_MIN_SILENCE_MS  = int(os.environ.get("VAD_MIN_SILENCE_MS",    "200"))
VAD_SAMPLE_RATE     = int(os.environ.get("VAD_SAMPLE_RATE",       "16000"))

# ── ONNX settings ──────────────────────────────────────────────────────────────
ONNX_THREADS = int(os.environ.get("ONNX_THREADS", "4"))

# ── FFmpeg settings ────────────────────────────────────────────────────────────
FFMPEG_TIMEOUT  = int(os.environ.get("FFMPEG_TIMEOUT",  "120"))
FFMPEG_THREADS  = int(os.environ.get("FFMPEG_THREADS",  "4"))
