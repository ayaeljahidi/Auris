"""Auris — Model singletons (lazy-loaded, thread-safe, CPU-optimized, VAD removed)"""
import logging
import threading
from pathlib import Path

import torch
from faster_whisper import WhisperModel
from transformers import T5ForConditionalGeneration, T5Tokenizer

from . import config

log = logging.getLogger("auris.models")

# ── Singleton state + per-model locks ─────────────────────────────────────────

_whisper_model:  WhisperModel | None               = None
_flan_model:     T5ForConditionalGeneration | None  = None
_flan_tokenizer: T5Tokenizer | None                = None

_whisper_lock = threading.Lock()
_flan_lock    = threading.Lock()


def load_whisper() -> WhisperModel:
    """Load faster-whisper optimized for CPU-only inference."""
    global _whisper_model
    if _whisper_model is not None:
        return _whisper_model
    with _whisper_lock:
        if _whisper_model is None:
            device  = "cpu"
            compute = "int8"  # int8 is fastest on CPU
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
    """
    Lazy-load Flan-T5 for the transcription correction layer.
    Returns (None, None) if FLAN_ENABLED=false in config.
    Thread-safe via double-checked locking.
    CPU-only, float32 for stability.
    """
    global _flan_model, _flan_tokenizer
    if not config.FLAN_ENABLED:
        return None, None
    if _flan_model is not None:
        return _flan_model, _flan_tokenizer
    with _flan_lock:
        if _flan_model is None:
            device = "cpu"
            log.info("Loading Flan-T5 (%s) on %s…", config.FLAN_MODEL, device.upper())
            _flan_tokenizer = T5Tokenizer.from_pretrained(
                config.FLAN_MODEL,
                cache_dir=config.FLAN_CACHE_DIR,
            )
            _flan_model = T5ForConditionalGeneration.from_pretrained(
                config.FLAN_MODEL,
                torch_dtype=torch.float32,  # float32 on CPU for stability
                device_map="cpu",
                cache_dir=config.FLAN_CACHE_DIR,
            )
            _flan_model.eval()
            log.info("✓ Flan-T5 loaded (%s | %s | float32)", config.FLAN_MODEL, device.upper())
    return _flan_model, _flan_tokenizer


def health_status() -> dict:
    return {
        "whisper_model":    config.WHISPER_MODEL,
        "flan_enabled":     config.FLAN_ENABLED,
        "flan_enabled_live": config.FLAN_ENABLED_LIVE,
        "flan_model":       config.FLAN_MODEL if config.FLAN_ENABLED else None,
        "device":           "cpu",
    }