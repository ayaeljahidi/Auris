"""Auris — Model singletons (Whisper + Flan + Wav2Vec2 Emotion + DistilRoBERTa Text Emotion)

P2 Optimisation: All four models are loaded concurrently at startup via
ThreadPoolExecutor instead of sequentially with await-to-thread chaining.

P3 Optimisation: Flan-T5 and Wav2Vec2 can be compiled with torch.compile when
TORCH_COMPILE=true.  Compilation is skipped on CPU by default because the
overhead typically outweighs the gain; enable only on GPU or with benchmarking.
"""
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import torch
from faster_whisper import WhisperModel
from transformers import (
    T5ForConditionalGeneration,
    T5Tokenizer,
    Wav2Vec2ForSequenceClassification,
    Wav2Vec2FeatureExtractor,
    pipeline as hf_pipeline,
)

from . import config

log = logging.getLogger("auris.models")

# ── Singleton state + per-model locks ─────────────────────────────────────────
_whisper_model:      WhisperModel | None = None
_flan_model:         T5ForConditionalGeneration | None = None
_flan_tokenizer:     T5Tokenizer | None = None
_emotion_model:      Wav2Vec2ForSequenceClassification | None = None
_emotion_processor:  Wav2Vec2FeatureExtractor | None = None
_emotion_labels:     list[str] = []
_text_emotion_pipe   = None   # DistilRoBERTa text emotion pipeline

_whisper_lock       = threading.Lock()
_flan_lock          = threading.Lock()
_emotion_lock       = threading.Lock()
_text_emotion_lock  = threading.Lock()


# ── 8-class Wav2Vec2 emotion labels ──────────────────────────────────────────
EMOTION_LABELS = [
    "angry", "calm", "disgust", "fear",
    "happy", "neutral", "sad", "surprised",
]


# ── Individual loaders ────────────────────────────────────────────────────────

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
            log.info(
                "✓ faster-whisper loaded (%s | %s | %s | threads=%d)",
                config.WHISPER_MODEL, device, compute, config.WHISPER_CPU_THREADS,
            )
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

            # P3: optional torch.compile
            if config.TORCH_COMPILE_ENABLED:
                try:
                    _flan_model = torch.compile(_flan_model, mode="reduce-overhead")
                    log.info("✓ Flan-T5 compiled with torch.compile (reduce-overhead)")
                except Exception as exc:
                    log.warning("torch.compile unavailable for Flan-T5: %s", exc)

            log.info("✓ Flan-T5 loaded (%s | CPU | float32)", config.FLAN_MODEL)
    return _flan_model, _flan_tokenizer


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

            # P3: optional torch.compile
            if config.TORCH_COMPILE_ENABLED:
                try:
                    _emotion_model = torch.compile(_emotion_model, mode="reduce-overhead")
                    log.info("✓ Wav2Vec2 compiled with torch.compile (reduce-overhead)")
                except Exception as exc:
                    log.warning("torch.compile unavailable for Wav2Vec2: %s", exc)

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
    """Public accessor for audio emotion model, processor, and labels."""
    model = load_emotion_model()
    return model, _emotion_processor, _emotion_labels


def load_text_emotion_model():
    """Load DistilRoBERTa text-emotion pipeline once (thread-safe)."""
    global _text_emotion_pipe

    if not config.TEXT_EMOTION_ENABLED:
        return None

    if _text_emotion_pipe is not None:
        return _text_emotion_pipe

    with _text_emotion_lock:
        if _text_emotion_pipe is not None:
            return _text_emotion_pipe

        log.info("Loading text emotion model: %s", config.TEXT_EMOTION_MODEL)
        try:
            _text_emotion_pipe = hf_pipeline(
                "text-classification",
                model=config.TEXT_EMOTION_MODEL,
                top_k=None,
                device=-1,
                torch_dtype=torch.float32,
            )
            log.info(
                "✓ Text emotion model loaded (%s | CPU | 7 classes)",
                config.TEXT_EMOTION_MODEL,
            )
        except Exception as exc:
            log.error("Failed to load text emotion model: %s", exc)
            _text_emotion_pipe = None

    return _text_emotion_pipe


# ── P2: Concurrent startup loader ─────────────────────────────────────────────

def load_all_models_concurrent() -> None:
    """
    Load all four models in parallel using a thread pool.

    Each loader is idempotent (double-checked locking), so calling them
    concurrently is safe.  On a 4-core machine this typically reduces total
    startup time from ~sequential_sum to ~max(individual_times).
    """
    loaders = {
        "whisper":       load_whisper,
        "flan":          load_flan,
        "wav2vec2":      load_emotion_model,
        "text_emotion":  load_text_emotion_model,
    }

    with ThreadPoolExecutor(max_workers=len(loaders), thread_name_prefix="model-load") as pool:
        futures = {pool.submit(fn): name for name, fn in loaders.items()}
        for future in as_completed(futures):
            name = futures[future]
            try:
                future.result()
                log.info("✓ [concurrent load] %s ready", name)
            except Exception as exc:
                log.error("✗ [concurrent load] %s failed: %s", name, exc)


# ── Health status ──────────────────────────────────────────────────────────────

def health_status() -> dict:
    return {
        "whisper_model":            config.WHISPER_MODEL,
        "whisper_loaded":           _whisper_model is not None,
        "flan_enabled":             config.FLAN_ENABLED,
        "flan_enabled_live":        config.FLAN_ENABLED_LIVE,
        "flan_model":               config.FLAN_MODEL if config.FLAN_ENABLED else None,
        "flan_loaded":              _flan_model is not None,
        # Audio emotion
        "emotion_enabled":          config.EMOTION_ENABLED,
        "emotion_model":            config.EMOTION_MODEL if config.EMOTION_ENABLED else None,
        "emotion_backend":          "wav2vec2" if _emotion_model else "none",
        "emotion_labels":           _emotion_labels if _emotion_labels else None,
        "emotion_num_classes":      len(_emotion_labels) if _emotion_labels else 0,
        # Text emotion
        "text_emotion_enabled":     config.TEXT_EMOTION_ENABLED,
        "text_emotion_model":       config.TEXT_EMOTION_MODEL if config.TEXT_EMOTION_ENABLED else None,
        "text_emotion_loaded":      _text_emotion_pipe is not None,
        # Fusion
        "fusion_audio_weight":      config.EMOTION_AUDIO_WEIGHT,
        "fusion_text_weight":       config.EMOTION_TEXT_WEIGHT,
        "device":                   "cpu",
        "torch_threads":            config.TORCH_THREADS,
        "torch_compile":            config.TORCH_COMPILE_ENABLED,
    }