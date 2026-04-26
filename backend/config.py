"""Auris — Configuration (aggressively CPU-optimized, VAD removed, FFmpeg-free)"""
import os

# ── Model paths ────────────────────────────────────────────────────────────────
WHISPER_MODEL   = os.environ.get("WHISPER_MODEL",    "base.en")
FLAN_MODEL      = os.environ.get("FLAN_MODEL",       "google/flan-t5-base")
FLAN_CACHE_DIR  = os.environ.get("FLAN_CACHE_DIR",   "models/flan-t5-base")

# ── Question Generation model (dedicated T5-QG, NOT Flan-T5) ───────────────────
QGEN_MODEL      = os.environ.get("QGEN_MODEL",       "valhalla/t5-base-qg-hl")
QGEN_CACHE_DIR  = os.environ.get("QGEN_CACHE_DIR",   "models/t5-base-qg-hl")

# ── Whisper settings (CPU-optimized) ───────────────────────────────────────────
WHISPER_BEAM_SIZE   = int(os.environ.get("WHISPER_BEAM_SIZE",   "1"))
WHISPER_LANGUAGE    = os.environ.get("WHISPER_LANGUAGE",        "en")
WHISPER_NUM_WORKERS = int(os.environ.get("WHISPER_NUM_WORKERS", "1"))
WHISPER_CPU_THREADS = int(os.environ.get("WHISPER_CPU_THREADS", "4"))

# ── ONNX settings ──────────────────────────────────────────────────────────────
ONNX_THREADS = int(os.environ.get("ONNX_THREADS", "4"))

# ── Flan-T5 correction layer (aggressively optimized) ──────────────────────────
FLAN_ENABLED         = os.environ.get("FLAN_ENABLED",         "true").lower() == "true"
FLAN_ENABLED_LIVE    = os.environ.get("FLAN_ENABLED_LIVE",    "false").lower() == "true"
FLAN_MAX_TOKENS      = int(os.environ.get("FLAN_MAX_TOKENS",   "32"))    # Correction: 32 tokens max
FLAN_NUM_BEAMS       = int(os.environ.get("FLAN_NUM_BEAMS",    "1"))     # Greedy

# ── Question generation (dedicated T5-QG model) ───────────────────────────────
QGEN_ENABLED         = os.environ.get("QGEN_ENABLED",         "true").lower() == "true"
QGEN_MAX_TOKENS      = int(os.environ.get("QGEN_MAX_TOKENS",   "48"))    # QG needs more tokens
QGEN_NUM_BEAMS       = int(os.environ.get("QGEN_NUM_BEAMS",    "1"))
QGEN_NUM_QUESTIONS   = int(os.environ.get("QGEN_NUM_QUESTIONS", "3"))     # How many questions to gen

# ── Critique thresholds ────────────────────────────────────────────────────────
CRITIQUE_NO_SPEECH_THRESHOLD   = float(os.environ.get("CRITIQUE_NO_SPEECH_THRESHOLD",   "0.5"))
CRITIQUE_AVG_LOGPROB_THRESHOLD = float(os.environ.get("CRITIQUE_AVG_LOGPROB_THRESHOLD", "-0.5"))
CRITIQUE_COMPRESSION_RATIO_MAX = float(os.environ.get("CRITIQUE_COMPRESSION_RATIO_MAX", "2.4"))

# ── WebSocket settings ─────────────────────────────────────────────────────────
WS_LIVE_TIMEOUT = int(os.environ.get("WS_LIVE_TIMEOUT", "300"))

# ── Audio extraction ───────────────────────────────────────────────────────────
PYAV_AUDIO_BUFFER_FRAMES = int(os.environ.get("PYAV_AUDIO_BUFFER_FRAMES", "4096"))