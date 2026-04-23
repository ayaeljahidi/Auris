"""Auris — Configuration (CPU-optimized, VAD removed, FFmpeg-free)"""
import os

# ── Model paths ────────────────────────────────────────────────────────────────
WHISPER_MODEL   = os.environ.get("WHISPER_MODEL",    "base.en")
FLAN_MODEL      = os.environ.get("FLAN_MODEL",       "google/flan-t5-base")
FLAN_CACHE_DIR  = os.environ.get("FLAN_CACHE_DIR",   "models/flan-t5-base")

# ── Whisper settings (CPU-optimized) ───────────────────────────────────────────
WHISPER_BEAM_SIZE   = int(os.environ.get("WHISPER_BEAM_SIZE",   "1"))
WHISPER_LANGUAGE    = os.environ.get("WHISPER_LANGUAGE",        "en")
WHISPER_NUM_WORKERS = int(os.environ.get("WHISPER_NUM_WORKERS", "1"))   # CPU: 1 worker optimal
WHISPER_CPU_THREADS = int(os.environ.get("WHISPER_CPU_THREADS", "4"))   # Match your CPU cores

# ── ONNX settings (kept for any future ONNX use) ───────────────────────────────
ONNX_THREADS = int(os.environ.get("ONNX_THREADS", "4"))

# ── Flan-T5 correction layer (aggressively optimized for CPU speed) ────────────
FLAN_ENABLED         = os.environ.get("FLAN_ENABLED",         "true").lower() == "true"
FLAN_ENABLED_LIVE    = os.environ.get("FLAN_ENABLED_LIVE",    "false").lower() == "true"  # Disabled for live — too slow
FLAN_MAX_TOKENS      = int(os.environ.get("FLAN_MAX_TOKENS",   "64"))    # Halved for speed
FLAN_NUM_BEAMS       = int(os.environ.get("FLAN_NUM_BEAMS",    "1"))     # Greedy — 2x faster

# ── Critique thresholds ────────────────────────────────────────────────────────
# Segments are critiqued before Flan-T5. If a segment looks "good enough",
# we skip Flan-T5 to save CPU cycles.
CRITIQUE_NO_SPEECH_THRESHOLD   = float(os.environ.get("CRITIQUE_NO_SPEECH_THRESHOLD",   "0.5"))
CRITIQUE_AVG_LOGPROB_THRESHOLD = float(os.environ.get("CRITIQUE_AVG_LOGPROB_THRESHOLD", "-0.5"))
CRITIQUE_COMPRESSION_RATIO_MAX = float(os.environ.get("CRITIQUE_COMPRESSION_RATIO_MAX", "2.4"))

# ── WebSocket settings ─────────────────────────────────────────────────────────
WS_LIVE_TIMEOUT = int(os.environ.get("WS_LIVE_TIMEOUT", "300"))