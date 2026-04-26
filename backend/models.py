"""Auris -- Model singletons (lazy-loaded, thread-safe, CPU-optimized)"""
import logging
import threading

import torch
from faster_whisper import WhisperModel
from transformers import T5ForConditionalGeneration, T5Tokenizer

from . import config

log = logging.getLogger("auris.models")

# -- Singleton state + per-model locks ----------------------------------------

_whisper_model:  WhisperModel | None               = None
_flan_model:     T5ForConditionalGeneration | None  = None
_flan_tokenizer: T5Tokenizer | None                = None
_qgen_model:     T5ForConditionalGeneration | None  = None
_qgen_tokenizer: T5Tokenizer | None                = None

_whisper_lock = threading.Lock()
_flan_lock    = threading.Lock()
_qgen_lock    = threading.Lock()


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
            log.info("faster-whisper loaded (%s | %s | %s | threads=%d)",
                     config.WHISPER_MODEL, device, compute, config.WHISPER_CPU_THREADS)
    return _whisper_model


def load_flan() -> tuple["T5ForConditionalGeneration", "T5Tokenizer"] | tuple[None, None]:
    """Lazy-load Flan-T5 for text correction. Returns (None, None) if FLAN_ENABLED=false."""
    global _flan_model, _flan_tokenizer
    if not config.FLAN_ENABLED:
        return None, None
    if _flan_model is not None:
        return _flan_model, _flan_tokenizer
    with _flan_lock:
        if _flan_model is None:
            device = "cpu"
            log.info("Loading Flan-T5 (%s) on %s...", config.FLAN_MODEL, device.upper())
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
            log.info("Flan-T5 loaded (%s | %s | float32)", config.FLAN_MODEL, device.upper())
    return _flan_model, _flan_tokenizer


def load_qgen() -> tuple["T5ForConditionalGeneration", "T5Tokenizer"] | tuple[None, None]:
    """Lazy-load dedicated T5-QG model for question generation."""
    global _qgen_model, _qgen_tokenizer
    if not config.QGEN_ENABLED:
        return None, None
    if _qgen_model is not None:
        return _qgen_model, _qgen_tokenizer
    with _qgen_lock:
        if _qgen_model is None:
            device = "cpu"
            log.info("Loading T5-QG (%s) on %s...", config.QGEN_MODEL, device.upper())
            _qgen_tokenizer = T5Tokenizer.from_pretrained(
                config.QGEN_MODEL,
                cache_dir=config.QGEN_CACHE_DIR,
            )
            _qgen_model = T5ForConditionalGeneration.from_pretrained(
                config.QGEN_MODEL,
                torch_dtype=torch.float32,
                device_map="cpu",
                cache_dir=config.QGEN_CACHE_DIR,
            )
            _qgen_model.eval()
            # Force full weight materialisation — prevents background download
            # continuing after this function returns (safetensors lazy-load issue)
            _dummy = _qgen_tokenizer("warmup", return_tensors="pt")
            with torch.no_grad():
                _qgen_model.generate(**_dummy, max_new_tokens=1)
            log.info("T5-QG loaded (%s | %s | float32)", config.QGEN_MODEL, device.upper())
    return _qgen_model, _qgen_tokenizer


def health_status() -> dict:
    return {
        "whisper_model":     config.WHISPER_MODEL,
        "flan_enabled":      config.FLAN_ENABLED,
        "flan_enabled_live": config.FLAN_ENABLED_LIVE,
        "flan_model":        config.FLAN_MODEL if config.FLAN_ENABLED else None,
        "qgen_enabled":      config.QGEN_ENABLED,
        "qgen_model":        config.QGEN_MODEL if config.QGEN_ENABLED else None,
        "device":            "cpu",
    }