"""Auris — Configuration (all environment variables)"""
import os
import torch

# ── Whisper settings ──────────────────────────────────────────────────────────
WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "base.en")
WHISPER_LANGUAGE = os.environ.get("WHISPER_LANGUAGE", "en")
WHISPER_BEAM_SIZE = int(os.environ.get("WHISPER_BEAM_SIZE", "5"))
WHISPER_NUM_WORKERS = int(os.environ.get("WHISPER_NUM_WORKERS", "1"))
WHISPER_CPU_THREADS = int(os.environ.get("WHISPER_CPU_THREADS", "4"))

# ── Flan-T5 correction layer ───────────────────────────────────────────────────
FLAN_ENABLED = os.environ.get("FLAN_ENABLED", "true").lower() == "true"
FLAN_ENABLED_LIVE = os.environ.get("FLAN_ENABLED_LIVE", "false").lower() == "true"
FLAN_MODEL = os.environ.get("FLAN_MODEL", "google/flan-t5-base")
FLAN_CACHE_DIR = os.environ.get("FLAN_CACHE_DIR", "models/flan-t5-base")
FLAN_MAX_TOKENS = int(os.environ.get("FLAN_MAX_TOKENS", "32"))
FLAN_NUM_BEAMS = int(os.environ.get("FLAN_NUM_BEAMS", "1"))

# ── Emotion detection (Wav2Vec2 - 8 classes) ─────────────────────────────────
EMOTION_ENABLED = os.environ.get("EMOTION_ENABLED", "true").lower() == "true"
EMOTION_MODEL = os.environ.get("EMOTION_MODEL", "prithivMLmods/Speech-Emotion-Classification")
EMOTION_SR = int(os.environ.get("EMOTION_SR", "16000"))

# ── Critique thresholds ────────────────────────────────────────────────────────
CRITIQUE_NO_SPEECH_THRESHOLD = float(os.environ.get("CRITIQUE_NO_SPEECH_THRESHOLD", "0.5"))
CRITIQUE_AVG_LOGPROB_THRESHOLD = float(os.environ.get("CRITIQUE_AVG_LOGPROB_THRESHOLD", "-0.5"))
CRITIQUE_COMPRESSION_RATIO_MAX = float(os.environ.get("CRITIQUE_COMPRESSION_RATIO_MAX", "2.4"))

# ── WebSocket settings ─────────────────────────────────────────────────────────
WS_LIVE_TIMEOUT = int(os.environ.get("WS_LIVE_TIMEOUT", "300"))

# ── Audio extraction ───────────────────────────────────────────────────────────
PYAV_AUDIO_BUFFER_FRAMES = int(os.environ.get("PYAV_AUDIO_BUFFER_FRAMES", "4096"))

# ── PyTorch global optimizations ───────────────────────────────────────────────
TORCH_THREADS = int(os.environ.get("TORCH_THREADS", "4"))
torch.set_num_threads(TORCH_THREADS)
torch.set_flush_denormal(True)
torch.set_grad_enabled(False)