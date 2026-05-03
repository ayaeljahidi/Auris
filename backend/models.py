"""Auris — Model singletons with ONNX Runtime (pre-converted community model)"""
import logging
import threading
import os
import time
import numpy as np
import torch
from faster_whisper import WhisperModel
from transformers import (
    T5ForConditionalGeneration, 
    T5Tokenizer, 
    AutoFeatureExtractor,
)

from . import config

log = logging.getLogger("auris.models")

# ── ONNX Runtime imports ──────────────────────────────────────────────────────
try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ort = None
    ONNX_AVAILABLE = False
    log.warning("ONNX Runtime not installed. Install with: pip install onnxruntime")

# ── Singleton state + per-model locks ─────────────────────────────────────────
_whisper_model:  WhisperModel | None = None
_flan_model:     T5ForConditionalGeneration | None = None
_flan_tokenizer: T5Tokenizer | None = None
_emotion_session: ort.InferenceSession | None = None
_emotion_extractor: AutoFeatureExtractor | None = None
_emotion_labels: list[str] = []

_whisper_lock  = threading.Lock()
_flan_lock     = threading.Lock()
_emotion_lock  = threading.Lock()


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


def _is_valid_onnx_file(filepath: str) -> bool:
    """Check if ONNX file exists and is valid (quick validation)."""
    if not os.path.exists(filepath):
        return False
    try:
        size = os.path.getsize(filepath)
        if size < 10 * 1024 * 1024:
            log.warning(f"ONNX file too small: {size} bytes (expected >10MB)")
            return False
    except Exception:
        return False
    return True


def _download_onnx_model_from_hf(cache_dir: str) -> str | None:
    """Download pre-converted ONNX model from Hugging Face Hub."""
    try:
        from huggingface_hub import hf_hub_download, list_repo_files

        log.info("Downloading ONNX model from HF: %s", config.EMOTION_MODEL)

        # List files in repo to find ONNX model
        files = list_repo_files(config.EMOTION_MODEL)
        onnx_files = [f for f in files if f.endswith('.onnx')]

        if not onnx_files:
            log.error("No ONNX files found in repo %s", config.EMOTION_MODEL)
            return None

        # Prefer model.onnx or the largest ONNX file
        target_file = 'model.onnx' if 'model.onnx' in onnx_files else onnx_files[0]

        downloaded_path = hf_hub_download(
            repo_id=config.EMOTION_MODEL,
            filename=target_file,
            cache_dir=cache_dir,
            local_dir=cache_dir,
            local_dir_use_symlinks=False,
        )

        log.info("✓ ONNX model downloaded: %s", downloaded_path)
        return downloaded_path

    except Exception as exc:
        log.error("Failed to download ONNX model: %s", exc)
        return None


def _load_labels_from_hf() -> list[str]:
    """Load emotion labels from the original model config."""
    global _emotion_labels

    try:
        from transformers import AutoConfig
        # Labels are in the original Dpngtm model, but we can infer from ONNX repo
        # or use the known 7-class mapping
        config_obj = AutoConfig.from_pretrained(
            config.EMOTION_MODEL,
            cache_dir=config.EMOTION_CACHE_DIR,
            trust_remote_code=True,
        )
        if hasattr(config_obj, 'id2label') and config_obj.id2label:
            _emotion_labels = [config_obj.id2label[i] for i in range(len(config_obj.id2label))]
            log.info("Labels loaded from HF config: %s", _emotion_labels)
            return _emotion_labels
    except Exception:
        pass

    # Fallback: known labels for this model (neutral replaces calm)
    _emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
    log.info("Using default labels: %s", _emotion_labels)
    return _emotion_labels


def load_emotion_model() -> tuple[ort.InferenceSession | None, AutoFeatureExtractor | None]:
    """
    Load pre-converted ONNX emotion model from Hugging Face.
    No PyTorch conversion needed — downloads ONNX directly.
    """
    global _emotion_session, _emotion_extractor, _emotion_labels

    if not config.EMOTION_ENABLED:
        return None, None

    if _emotion_session is not None:
        return _emotion_session, _emotion_extractor

    with _emotion_lock:
        if _emotion_session is not None:
            return _emotion_session, _emotion_extractor

        log.info("Loading emotion ONNX model: %s", config.EMOTION_MODEL)

        try:
            # Ensure cache directory exists
            os.makedirs(config.EMOTION_CACHE_DIR, exist_ok=True)

            # Load feature extractor from the ONNX repo (has preprocessor_config.json)
            _emotion_extractor = AutoFeatureExtractor.from_pretrained(
                config.EMOTION_MODEL,
                cache_dir=config.EMOTION_CACHE_DIR,
                trust_remote_code=True,
            )

            # Download ONNX model if not cached
            onnx_path = os.path.join(config.EMOTION_CACHE_DIR, "model.onnx")

            if not _is_valid_onnx_file(onnx_path):
                log.info("ONNX model not found locally — downloading from Hugging Face...")
                downloaded = _download_onnx_model_from_hf(config.EMOTION_CACHE_DIR)
                if downloaded and downloaded != onnx_path:
                    # Copy/symlink to expected location
                    import shutil
                    shutil.copy2(downloaded, onnx_path)
                    log.info("✓ ONNX model cached at: %s", onnx_path)
            else:
                log.info("✓ ONNX model found in cache — loading directly")

            # Load labels
            _load_labels_from_hf()

            # Configure ONNX Runtime session
            sess_options = ort.SessionOptions()
            sess_options.enable_profiling = False
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            sess_options.intra_op_num_threads = config.ONNX_THREADS
            sess_options.inter_op_num_threads = 1
            sess_options.add_session_config_entry("session.intra_op.use_prepacked_functions", "1")

            providers = ['CPUExecutionProvider']

            _emotion_session = ort.InferenceSession(
                onnx_path,
                sess_options=sess_options,
                providers=providers
            )

            # Log input/output info for debugging
            inputs = _emotion_session.get_inputs()
            outputs = _emotion_session.get_outputs()
            log.info("ONNX model I/O — inputs: %s, outputs: %s",
                     [i.name for i in inputs], [o.name for o in outputs])

            log.info("✓ ONNX Emotion model loaded (CPU | threads=%d | %d labels)", 
                     config.ONNX_THREADS, len(_emotion_labels))
            log.info("  Labels: %s", _emotion_labels)

            # Quick validation
            test_input = np.random.randn(1, config.EMOTION_SR * 2).astype(np.float32)
            try:
                input_name = inputs[0].name
                test_output = _emotion_session.run(None, {input_name: test_input})
                log.info("✓ ONNX session validated (output shape: %s)", test_output[0].shape)
            except Exception as e:
                log.warning(f"ONNX validation warning: {e}")

        except Exception as exc:
            log.error("Failed to load emotion model: %s", exc)
            return None, None

    return _emotion_session, _emotion_extractor


def get_emotion_session():
    """Public accessor for persistent ONNX session."""
    session, _ = load_emotion_model()
    return session, _emotion_labels


def health_status() -> dict:
    return {
        "whisper_model":       config.WHISPER_MODEL,
        "flan_enabled":        config.FLAN_ENABLED,
        "flan_enabled_live":   config.FLAN_ENABLED_LIVE,
        "flan_model":          config.FLAN_MODEL if config.FLAN_ENABLED else None,
        "emotion_enabled":     config.EMOTION_ENABLED,
        "emotion_model":       config.EMOTION_MODEL if config.EMOTION_ENABLED else None,
        "emotion_backend":     "onnx" if _emotion_session else "none",
        "emotion_labels":      _emotion_labels if _emotion_labels else None,
        "device":              "cpu",
        "onnx_threads":        config.ONNX_THREADS,
        "emotion_parallel_workers": config.EMOTION_PARALLEL_WORKERS,
    }