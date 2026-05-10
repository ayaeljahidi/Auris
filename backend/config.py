"""Auris — Configuration (all environment variables)"""
import os
import threading
import torch

# ── Whisper settings ──────────────────────────────────────────────────────────
WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "base.en")
WHISPER_LANGUAGE = os.environ.get("WHISPER_LANGUAGE", "en")
WHISPER_BEAM_SIZE = int(os.environ.get("WHISPER_BEAM_SIZE", "5"))
WHISPER_NUM_WORKERS = int(os.environ.get("WHISPER_NUM_WORKERS", "1"))
WHISPER_CPU_THREADS = int(os.environ.get("WHISPER_CPU_THREADS", "4"))

# ── Flan-T5 correction layer ──────────────────────────────────────────────────
FLAN_ENABLED = os.environ.get("FLAN_ENABLED", "true").lower() == "true"
FLAN_ENABLED_LIVE = os.environ.get("FLAN_ENABLED_LIVE", "true").lower() == "true"
FLAN_MODEL = os.environ.get("FLAN_MODEL", "google/flan-t5-base")
FLAN_CACHE_DIR = os.environ.get("FLAN_CACHE_DIR", "models/flan-t5-base")
FLAN_MAX_TOKENS = int(os.environ.get("FLAN_MAX_TOKENS", "32"))
FLAN_NUM_BEAMS = int(os.environ.get("FLAN_NUM_BEAMS", "1"))
# P2: Token-based chunking — max tokens per Flan segment (real token budget)
FLAN_MAX_INPUT_TOKENS = int(os.environ.get("FLAN_MAX_INPUT_TOKENS", "128"))

# ── Audio emotion detection (Wav2Vec2 — 8 classes) ────────────────────────────
EMOTION_ENABLED = os.environ.get("EMOTION_ENABLED", "true").lower() == "true"
EMOTION_MODEL   = os.environ.get("EMOTION_MODEL", "prithivMLmods/Speech-Emotion-Classification")
EMOTION_SR      = int(os.environ.get("EMOTION_SR", "16000"))

# ── Text emotion detection (DistilRoBERTa — 7 classes) ───────────────────────
TEXT_EMOTION_ENABLED = os.environ.get("TEXT_EMOTION_ENABLED", "true").lower() == "true"
TEXT_EMOTION_MODEL   = os.environ.get(
    "TEXT_EMOTION_MODEL",
    "j-hartmann/emotion-english-distilroberta-base",
)

# ── Emotion fusion weights ─────────────────────────────────────────────────────
# Default weights used when neither signal is strongly dominant.
# Adaptive rules in emotion_fusion.py may override them automatically.
EMOTION_AUDIO_WEIGHT = float(os.environ.get("EMOTION_AUDIO_WEIGHT", "0.40"))
EMOTION_TEXT_WEIGHT  = float(os.environ.get("EMOTION_TEXT_WEIGHT",  "0.60"))

# ── Critique thresholds ────────────────────────────────────────────────────────
CRITIQUE_NO_SPEECH_THRESHOLD    = float(os.environ.get("CRITIQUE_NO_SPEECH_THRESHOLD", "0.5"))
CRITIQUE_AVG_LOGPROB_THRESHOLD  = float(os.environ.get("CRITIQUE_AVG_LOGPROB_THRESHOLD", "-0.5"))
CRITIQUE_COMPRESSION_RATIO_MAX  = float(os.environ.get("CRITIQUE_COMPRESSION_RATIO_MAX", "2.4"))

# ── WebSocket settings ─────────────────────────────────────────────────────────
WS_LIVE_TIMEOUT = int(os.environ.get("WS_LIVE_TIMEOUT", "300"))
# P3: WebSocket backpressure — max queued send bytes before we start dropping partials
WS_SEND_QUEUE_MAX_BYTES = int(os.environ.get("WS_SEND_QUEUE_MAX_BYTES", str(512 * 1024)))

# ── Audio extraction ──────────────────────────────────────────────────────────
# P2: Pre-allocate numpy audio buffer to this many samples (16kHz × seconds).
# 0 = dynamic allocation (legacy).  300s × 16000 = 4_800_000 samples ≈ 18 MB.
PYAV_AUDIO_BUFFER_FRAMES   = int(os.environ.get("PYAV_AUDIO_BUFFER_FRAMES", "4096"))
AUDIO_PREALLOCATE_SAMPLES  = int(os.environ.get("AUDIO_PREALLOCATE_SAMPLES", str(16000 * 300)))

# ── PyTorch global optimisations ──────────────────────────────────────────────
TORCH_THREADS = int(os.environ.get("TORCH_THREADS", "0"))  # 0 = PyTorch auto-pick
_cpu_count = os.cpu_count() or 4
_intra = TORCH_THREADS if TORCH_THREADS > 0 else _cpu_count
_inter = max(2, _cpu_count // 2)
torch.set_num_threads(_intra)
torch.set_num_interop_threads(_inter)
torch.set_flush_denormal(True)
torch.set_grad_enabled(False)

# P3: torch.compile — set TORCH_COMPILE=true to opt in (requires PyTorch ≥ 2.0)
TORCH_COMPILE_ENABLED = os.environ.get("TORCH_COMPILE", "false").lower() == "true"

# ── P1: Config-mutation lock ──────────────────────────────────────────────────
# Replacing bare attribute assignments with a lock-protected context manager
# eliminates the race condition in _live_pipeline where FLAN_ENABLED was
# temporarily mutated across threads.
_config_lock = threading.Lock()


class _FlanLiveOverride:
    """Context manager that safely overrides FLAN_ENABLED for the live pipeline."""

    def __init__(self, value: bool):
        self._value = value
        self._saved: bool | None = None

    def __enter__(self) -> None:
        global FLAN_ENABLED
        with _config_lock:
            self._saved = FLAN_ENABLED
            FLAN_ENABLED = self._value

    def __exit__(self, *_) -> None:
        global FLAN_ENABLED
        with _config_lock:
            FLAN_ENABLED = self._saved  # type: ignore[assignment]


def flan_live_override(value: bool) -> _FlanLiveOverride:
    """Return a context manager that sets FLAN_ENABLED=value for its duration."""
    return _FlanLiveOverride(value)