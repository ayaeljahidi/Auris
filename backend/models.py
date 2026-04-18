"""Vosper — Model singletons (lazy-loaded, thread-safe via GIL)"""
import logging
from pathlib import Path

import onnxruntime as ort
import torch
import vosk
from faster_whisper import WhisperModel

from . import config

log = logging.getLogger("vosper.models")

_vosk_model:        vosk.Model | None      = None
_whisper_model:     WhisperModel | None    = None
_marblenet_session: ort.InferenceSession | None = None


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


def health_status() -> dict:
    return {
        "vosk_ready":      Path(config.VOSK_PATH).exists(),
        "whisper_model":   config.WHISPER_MODEL,
        "marblenet_ready": Path(config.MARBLENET_PATH).exists(),
    }
