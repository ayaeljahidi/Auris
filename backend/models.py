"""Vosper — Model singletons (lazy-loaded, thread-safe via GIL)"""
import logging
from pathlib import Path

import onnxruntime as ort
import torch
import vosk
from faster_whisper import WhisperModel
from transformers import T5ForConditionalGeneration, T5Tokenizer

from . import config

log = logging.getLogger("vosper.models")

_vosk_model:        vosk.Model | None               = None
_whisper_model:     WhisperModel | None             = None
_marblenet_session: ort.InferenceSession | None     = None
_flan_model:        T5ForConditionalGeneration | None = None
_flan_tokenizer:    T5Tokenizer | None              = None


def load_vosk() -> vosk.Model:
    global _vosk_model
    if _vosk_model is None:
        if not Path(config.VOSK_PATH).exists():
            raise RuntimeError(
                f"Vosk model not found at '{config.VOSK_PATH}'. "
                "Run: python scripts/setup_models.py"
            )
        vosk.SetLogLevel(-1)
        _vosk_model = vosk.Model(config.VOSK_PATH)
        log.info("✓ Vosk model loaded")
    return _vosk_model


def load_whisper() -> WhisperModel:
    global _whisper_model
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
        log.info("✓ faster-whisper loaded (%s | %s | %s)", config.WHISPER_MODEL, device, compute)
    return _whisper_model


def load_marblenet() -> ort.InferenceSession | None:
    global _marblenet_session
    if _marblenet_session is None:
        if not Path(config.MARBLENET_PATH).exists():
            log.warning("MarbleNet not found at '%s' — falling back to pass-through VAD", config.MARBLENET_PATH)
            return None
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.intra_op_num_threads      = config.ONNX_THREADS
        opts.enable_mem_pattern        = True
        _marblenet_session = ort.InferenceSession(
            config.MARBLENET_PATH, opts, providers=["CPUExecutionProvider"]
        )
        log.info("✓ MarbleNet VAD loaded (ONNX)")
    return _marblenet_session


def load_flan() -> tuple["T5ForConditionalGeneration", "T5Tokenizer"] | tuple[None, None]:
    """
    Lazy-load Flan-T5 for the transcription correction layer.
    Returns (None, None) if FLAN_ENABLED=false in config.
    """
    global _flan_model, _flan_tokenizer
    if not config.FLAN_ENABLED:
        return None, None
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
        "vosk_ready":      Path(config.VOSK_PATH).exists(),
        "whisper_model":   config.WHISPER_MODEL,
        "marblenet_ready": Path(config.MARBLENET_PATH).exists(),
        "flan_enabled":    config.FLAN_ENABLED,
        "flan_model":      config.FLAN_MODEL if config.FLAN_ENABLED else None,
    }