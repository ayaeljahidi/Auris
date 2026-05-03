"""Auris — Configuration (aggressively CPU-optimized with ONNX)"""
import os
import torch

# ── Model paths ────────────────────────────────────────────────────────────────
WHISPER_MODEL   = os.environ.get("WHISPER_MODEL",    "base.en")
FLAN_MODEL      = os.environ.get("FLAN_MODEL",       "google/flan-t5-base")
FLAN_CACHE_DIR  = os.environ.get("FLAN_CACHE_DIR",   "models/flan-t5-base")

# ── Emotion detection with ONNX (pre-converted community model) ───────────────
EMOTION_MODEL     = os.environ.get("EMOTION_MODEL",     "onnx-community/wav2vec2-emotion-recognition-ONNX")
EMOTION_CACHE_DIR = os.environ.get("EMOTION_CACHE_DIR", "models/wav2vec2-emotion-onnx")
EMOTION_ENABLED   = os.environ.get("EMOTION_ENABLED",   "true").lower() == "true"
EMOTION_MIN_DURATION = float(os.environ.get("EMOTION_MIN_DURATION", "1.0"))
EMOTION_SR = 16_000

# ── Emotion chunking (optimized for speed with parallel processing) ────────────
EMOTION_CHUNK_SECONDS    = float(os.environ.get("EMOTION_CHUNK_SECONDS",    "10.0"))
EMOTION_OVERLAP_SECONDS  = float(os.environ.get("EMOTION_OVERLAP_SECONDS",  "0.0"))
EMOTION_MIN_CHUNK_SECONDS = float(os.environ.get("EMOTION_MIN_CHUNK_SECONDS", "1.0"))
EMOTION_PARALLEL_WORKERS  = int(os.environ.get("EMOTION_PARALLEL_WORKERS",  "4"))

# ── ONNX Runtime optimizations ─────────────────────────────────────────────────
ONNX_THREADS = int(os.environ.get("ONNX_THREADS", "4"))
ONNX_USE_FP16 = os.environ.get("ONNX_USE_FP16", "true").lower() == "true"

# ── Whisper settings (CPU-optimized) ───────────────────────────────────────────
WHISPER_BEAM_SIZE   = int(os.environ.get("WHISPER_BEAM_SIZE",   "1"))
WHISPER_LANGUAGE    = os.environ.get("WHISPER_LANGUAGE",        "en")
WHISPER_NUM_WORKERS = int(os.environ.get("WHISPER_NUM_WORKERS", "1"))
WHISPER_CPU_THREADS = int(os.environ.get("WHISPER_CPU_THREADS", "4"))

# ── Flan-T5 correction layer (aggressively optimized) ──────────────────────────
FLAN_ENABLED         = os.environ.get("FLAN_ENABLED",         "true").lower() == "true"
FLAN_ENABLED_LIVE    = os.environ.get("FLAN_ENABLED_LIVE",    "false").lower() == "true"
FLAN_MAX_TOKENS      = int(os.environ.get("FLAN_MAX_TOKENS",   "32"))
FLAN_NUM_BEAMS       = int(os.environ.get("FLAN_NUM_BEAMS",    "1"))

# ── Critique thresholds ────────────────────────────────────────────────────────
CRITIQUE_NO_SPEECH_THRESHOLD   = float(os.environ.get("CRITIQUE_NO_SPEECH_THRESHOLD",   "0.5"))
CRITIQUE_AVG_LOGPROB_THRESHOLD = float(os.environ.get("CRITIQUE_AVG_LOGPROB_THRESHOLD", "-0.5"))
CRITIQUE_COMPRESSION_RATIO_MAX = float(os.environ.get("CRITIQUE_COMPRESSION_RATIO_MAX", "2.4"))

# ── WebSocket settings ─────────────────────────────────────────────────────────
WS_LIVE_TIMEOUT = int(os.environ.get("WS_LIVE_TIMEOUT", "300"))

# ── Audio extraction ───────────────────────────────────────────────────────────
PYAV_AUDIO_BUFFER_FRAMES = int(os.environ.get("PYAV_AUDIO_BUFFER_FRAMES", "4096"))

# ── PyTorch global optimizations ───────────────────────────────────────────────
torch.set_num_threads(ONNX_THREADS)
torch.set_flush_denormal(True)
torch.set_grad_enabled(False)