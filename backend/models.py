"""Auris — Model singletons (Whisper + Flan + Wav2Vec2 Emotion)"""
import logging
import threading
import numpy as np
import torch
from faster_whisper import WhisperModel
from transformers import (
    T5ForConditionalGeneration,
    T5Tokenizer,
    Wav2Vec2ForSequenceClassification,
    Wav2Vec2FeatureExtractor,
)

from . import config

log = logging.getLogger("auris.models")

# ── Singleton state + per-model locks ─────────────────────────────────────────
_whisper_model:  WhisperModel | None = None
_flan_model:     T5ForConditionalGeneration | None = None
_flan_tokenizer: T5Tokenizer | None = None
_emotion_model:  Wav2Vec2ForSequenceClassification | None = None
_emotion_processor: Wav2Vec2FeatureExtractor | None = None
_emotion_labels: list[str] = []

_whisper_lock  = threading.Lock()
_flan_lock     = threading.Lock()
_emotion_lock  = threading.Lock()


# ── 8-class Wav2Vec2 emotion labels (prithivMLmods/Speech-Emotion-Classification)
EMOTION_LABELS = [
    "angry", "calm", "disgust", "fear",
    "happy", "neutral", "sad", "surprised",
]


def load_whisper() -> WhisperModel:
    """Load faster-whisper optimized for CPU-only inference."""
    global _whisper_model
    if _whisper_model is not None:
        return _whisper_model
    with _whisper_lock:
        if _whisper_model is None:
            device  = "cpu"
            compute = "int8"
            _whisper_model = WhisperModel(
                config.WHISPER_MODEL,
                device=device,
                compute_type=compute,
                num_workers=config.WHISPER_NUM_WORKERS,
                cpu_threads=config.WHISPER_CPU_THREADS,
            )
            log.info("✓ faster-whisper loaded (%s | %s | %s | threads=%d)",
                     config.WHISPER_MODEL, device, compute, config.WHISPER_CPU_THREADS)
    return _whisper_model


def load_flan() -> tuple["T5ForConditionalGeneration", "T5Tokenizer"] | tuple[None, None]:
    """Lazy-load Flan-T5. Returns (None, None) if FLAN_ENABLED=false."""
    global _flan_model, _flan_tokenizer
    if not config.FLAN_ENABLED:
        return None, None
    if _flan_model is not None:
        return _flan_model, _flan_tokenizer
    with _flan_lock:
        if _flan_model is None:
            log.info("Loading Flan-T5 (%s)…", config.FLAN_MODEL)
            _flan_tokenizer = T5Tokenizer.from_pretrained(
                config.FLAN_MODEL,
                cache_dir=config.FLAN_CACHE_DIR,
            )
            _flan_model = T5ForConditionalGeneration.from_pretrained(
                config.FLAN_MODEL,
                torch_dtype=torch.float32,
                device_map="cpu",
                cache_dir=config.FLAN_CACHE_DIR,
            )
            _flan_model.eval()
            log.info("✓ Flan-T5 loaded (%s | CPU | float32)", config.FLAN_MODEL)
    return _flan_model, _flan_tokenizer


# ── Wav2Vec2 Emotion loader (standard Transformers, no compilation) ───────────

def load_emotion_model() -> Wav2Vec2ForSequenceClassification | None:
    """Load Wav2Vec2 emotion model (CPU, standard Transformers API)."""
    global _emotion_model, _emotion_processor, _emotion_labels

    if not config.EMOTION_ENABLED:
        return None

    if _emotion_model is not None:
        return _emotion_model

    with _emotion_lock:
        if _emotion_model is not None:
            return _emotion_model

        log.info("Loading Wav2Vec2 emotion model: %s", config.EMOTION_MODEL)

        try:
            _emotion_processor = Wav2Vec2FeatureExtractor.from_pretrained(
                config.EMOTION_MODEL,
            )
            _emotion_model = Wav2Vec2ForSequenceClassification.from_pretrained(
                config.EMOTION_MODEL,
                torch_dtype=torch.float32,
            )
            _emotion_model.eval()

            # Use model\'s actual label count (should be 8)
            num_labels = _emotion_model.config.num_labels
            _emotion_labels = EMOTION_LABELS[:num_labels]

            log.info("✓ Wav2Vec2 emotion loaded (CPU | %d classes | ~378MB)", num_labels)
            log.info("  Labels: %s", _emotion_labels)

            # Quick validation
            with torch.no_grad():
                dummy = torch.zeros(1, config.EMOTION_SR * 2).float()
                inputs = _emotion_processor(
                    dummy.squeeze().numpy(),
                    sampling_rate=config.EMOTION_SR,
                    return_tensors="pt",
                )
                test_out = _emotion_model(**inputs)
                log.info("✓ Validation passed (logits shape: %s)", test_out.logits.shape)

        except Exception as exc:
            log.error("Failed to load Wav2Vec2 emotion model: %s", exc)
            _emotion_model = None
            _emotion_processor = None
            return None

    return _emotion_model


def get_emotion_session() -> tuple:
    """Public accessor for emotion model, processor, and labels."""
    model = load_emotion_model()
    return model, _emotion_processor, _emotion_labels


def health_status() -> dict:
    return {
        "whisper_model":       config.WHISPER_MODEL,
        "whisper_loaded":      _whisper_model is not None,
        "flan_enabled":        config.FLAN_ENABLED,
        "flan_enabled_live":   config.FLAN_ENABLED_LIVE,
        "flan_model":          config.FLAN_MODEL if config.FLAN_ENABLED else None,
        "flan_loaded":         _flan_model is not None,
        "emotion_enabled":     config.EMOTION_ENABLED,
        "emotion_model":       config.EMOTION_MODEL if config.EMOTION_ENABLED else None,
        "emotion_backend":     "wav2vec2" if _emotion_model else "none",
        "emotion_labels":      _emotion_labels if _emotion_labels else None,
        "emotion_num_classes": len(_emotion_labels) if _emotion_labels else 0,
        "device":              "cpu",
        "torch_threads":       config.TORCH_THREADS,
    }