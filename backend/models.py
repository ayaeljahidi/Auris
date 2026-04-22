"""Auris — Model singletons (lazy-loaded, thread-safe)"""
import logging
import threading
from pathlib import Path

import onnxruntime as ort
import torch
from faster_whisper import WhisperModel
from transformers import T5ForConditionalGeneration, T5Tokenizer

from . import config

log = logging.getLogger("auris.models")

# ── Singleton state + per-model locks ─────────────────────────────────────────

_whisper_model:  WhisperModel | None               = None
_silero_session: ort.InferenceSession | None        = None
_flan_model:     T5ForConditionalGeneration | None  = None
_flan_tokenizer: T5Tokenizer | None                = None

_whisper_lock = threading.Lock()
_silero_lock  = threading.Lock()
_flan_lock    = threading.Lock()


def load_whisper() -> WhisperModel:
    """Load faster-whisper (singleton, thread-safe via double-checked locking)."""
    global _whisper_model
    if _whisper_model is not None:
        return _whisper_model
    with _whisper_lock:
        if _whisper_model is None:
            device  = "cuda" if torch.cuda.is_available() else "cpu"
            compute = "int8" if device == "cpu" else "float16"
            _whisper_model = WhisperModel(
                config.WHISPER_MODEL,
                device=device,
                compute_type=compute,
                num_workers=config.WHISPER_NUM_WORKERS,
                cpu_threads=config.WHISPER_CPU_THREADS,
            )
            log.info("✓ faster-whisper loaded (%s | %s | %s)",
                     config.WHISPER_MODEL, device, compute)
    return _whisper_model


def load_silero() -> ort.InferenceSession | None:
    """Load the Silero VAD ONNX session (singleton, thread-safe)."""
    global _silero_session
    if _silero_session is not None:
        return _silero_session
    with _silero_lock:
        if _silero_session is None:
            if not Path(config.SILERO_PATH).exists():
                log.warning(
                    "Silero VAD not found at '%s' — falling back to pass-through VAD",
                    config.SILERO_PATH,
                )
                return None
            opts = ort.SessionOptions()
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            opts.intra_op_num_threads     = config.ONNX_THREADS
            opts.enable_mem_pattern       = True
            _silero_session = ort.InferenceSession(
                config.SILERO_PATH, opts, providers=["CPUExecutionProvider"]
            )
            log.info("✓ Silero VAD loaded (ONNX)")
    return _silero_session


def load_flan() -> tuple["T5ForConditionalGeneration", "T5Tokenizer"] | tuple[None, None]:
    """
    Lazy-load Flan-T5 for the transcription correction layer.
    Returns (None, None) if FLAN_ENABLED=false in config.
    Thread-safe via double-checked locking.
    """
    global _flan_model, _flan_tokenizer
    if not config.FLAN_ENABLED:
        return None, None
    if _flan_model is not None:
        return _flan_model, _flan_tokenizer
    with _flan_lock:
        if _flan_model is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            log.info("Loading Flan-T5 (%s) on %s…", config.FLAN_MODEL, device.upper())
            _flan_tokenizer = T5Tokenizer.from_pretrained(config.FLAN_MODEL)
            _flan_model = T5ForConditionalGeneration.from_pretrained(
                config.FLAN_MODEL,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            ).to(device)
            _flan_model.eval()
            log.info("✓ Flan-T5 loaded (%s | %s)", config.FLAN_MODEL, device.upper())
    return _flan_model, _flan_tokenizer


def health_status() -> dict:
    return {
        "whisper_model":    config.WHISPER_MODEL,
        "silero_vad_ready": Path(config.SILERO_PATH).exists(),
        "flan_enabled":     config.FLAN_ENABLED,
        "flan_model":       config.FLAN_MODEL if config.FLAN_ENABLED else None,
    }