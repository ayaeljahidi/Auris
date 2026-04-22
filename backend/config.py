"""Auris — Configuration"""
import os

# ── Model paths ────────────────────────────────────────────────────────────────
WHISPER_MODEL   = os.environ.get("WHISPER_MODEL",    "base.en")
SILERO_PATH     = os.environ.get("SILERO_PATH",      "models/silero_vad/silero_vad.onnx")

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

# ── Flan-T5 correction layer ───────────────────────────────────────────────────
# Set FLAN_ENABLED=false to disable the correction layer entirely.
# Set FLAN_MODEL to "google/flan-t5-large" for higher quality (slower).
FLAN_ENABLED    = os.environ.get("FLAN_ENABLED",  "true").lower() == "true"
FLAN_MODEL      = os.environ.get("FLAN_MODEL",    "google/flan-t5-base")
FLAN_MAX_TOKENS = int(os.environ.get("FLAN_MAX_TOKENS", "512"))
FLAN_NUM_BEAMS  = int(os.environ.get("FLAN_NUM_BEAMS",  "4"))

# ── WebSocket settings ─────────────────────────────────────────────────────────
# WS_LIVE_TIMEOUT: seconds before an idle live WebSocket is closed automatically.
WS_LIVE_TIMEOUT = int(os.environ.get("WS_LIVE_TIMEOUT", "300"))