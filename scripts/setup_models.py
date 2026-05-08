"""Auris — Model Setup Script (CPU-only, no compilation, Wav2Vec2 emotion)"""
import os
import sys
import subprocess
import platform
from pathlib import Path

# ── Constants ──────────────────────────────────────────────────────────────────
FLAN_MODEL = os.environ.get("FLAN_MODEL", "google/flan-t5-base")
FLAN_DIR = Path("models/flan-t5-base")
WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "base.en")

# New emotion model (Wav2Vec2 - no compilation)
EMOTION_MODEL = os.environ.get("EMOTION_MODEL", "prithivMLmods/Speech-Emotion-Classification")

OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:1.5b")
OLLAMA_URL = "http://localhost:11434"

LINE = "-" * 60


# ── Printer helpers ────────────────────────────────────────────────────────────
def header(text: str) -> None:
    print()
    print(LINE)
    print("  " + text)
    print(LINE)


def step(n: int, total: int, text: str) -> None:
    print()
    print(f"[{n}/{total}] {text}")


def ok(msg: str) -> None:
    print(f"  [OK] {msg}")


def warn(msg: str) -> None:
    print(f"  [WARN] {msg}")


def err(msg: str) -> None:
    print(f"  [ERR] {msg}")


def check_python() -> None:
    if sys.version_info < (3, 10):
        err("Python 3.10+ is required.")
        sys.exit(1)
    ok(f"Python {sys.version_info.major}.{sys.version_info.minor}")


def check_packages() -> None:
    required = [
        "fastapi", "uvicorn", "av", "numpy", "torch", "transformers",
        "sentencepiece", "huggingface_hub", "librosa", "soundfile"
    ]
    missing = []
    for pkg in required:
        try:
            __import__(pkg)
            ok(pkg)
        except ImportError:
            err(pkg)
            missing.append(pkg)
    if missing:
        print("\n  Missing packages. Install with:")
        print("    pip install -r requirements.txt")
        sys.exit(1)


def download_whisper() -> None:
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        err("faster-whisper not installed")
        sys.exit(1)

    print(f"  Downloading faster-whisper {WHISPER_MODEL}...")
    try:
        _ = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8", cpu_threads=4)
        ok(f"faster-whisper ({WHISPER_MODEL} / int8 / CPU) ready")
    except Exception as exc:
        err(f"Whisper download failed: {exc}")
        sys.exit(1)


def download_flan() -> None:
    try:
        from transformers import T5ForConditionalGeneration, T5Tokenizer
    except ImportError:
        err("transformers not installed")
        sys.exit(1)

    FLAN_DIR.mkdir(parents=True, exist_ok=True)
    print(f"  Downloading {FLAN_MODEL} (~1 GB)...")
    try:
        _ = T5Tokenizer.from_pretrained(FLAN_MODEL, cache_dir=str(FLAN_DIR))
        _ = T5ForConditionalGeneration.from_pretrained(
            FLAN_MODEL,
            torch_dtype="auto",
            device_map="cpu",
            cache_dir=str(FLAN_DIR),
        )
        ok(f"Flan-T5 ({FLAN_MODEL}) downloaded")
    except Exception as exc:
        err(f"Flan-T5 download failed: {exc}")
        sys.exit(1)


def download_emotion_model() -> None:
    """Download Wav2Vec2 emotion model - no compilation needed."""
    try:
        from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor
    except ImportError:
        err("transformers not installed")
        sys.exit(1)

    print(f"  Downloading Wav2Vec2 emotion model: {EMOTION_MODEL}")
    print("  Emotions: Anger, Calm, Disgust, Fear, Happy, Neutral, Sad, Surprised")
    print("  (~378MB, downloads once and caches locally)")

    try:
        # Download model
        model = Wav2Vec2ForSequenceClassification.from_pretrained(EMOTION_MODEL)
        processor = Wav2Vec2FeatureExtractor.from_pretrained(EMOTION_MODEL)
        
        # Get number of labels safely
        num_labels = model.config.num_labels if hasattr(model.config, 'num_labels') else 8
        labels = model.config.id2label if hasattr(model.config, 'id2label') else None
        
        # Quick test with dummy audio
        import torch
        import numpy as np
        dummy = np.zeros(16000, dtype=np.float32)
        inputs = processor(dummy, sampling_rate=16000, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
        
        ok(f"Wav2Vec2 emotion model ready - {num_labels} emotion classes")
        if labels:
            ok(f"  Labels: {list(labels.values())}")
        
    except Exception as exc:
        err(f"Emotion model download failed: {exc}")
        print("\n  [WARN] Emotion detection will be disabled. Continue anyway? (y/n)")
        if input().lower() != "y":
            sys.exit(1)


def setup_ollama_qg() -> None:
    """Ollama setup - unchanged from original."""
    print()
    print(f"  QG model : {OLLAMA_MODEL}")
    print(f"  Endpoint : {OLLAMA_URL}")
    print()

    # Check if Ollama is installed
    ollama_on_path = subprocess.run(
        ["ollama", "--version"],
        capture_output=True,
    ).returncode == 0

    if not ollama_on_path:
        print("  Ollama not found. Please install from: https://ollama.com")
        print("  Continue without QG? (y/n)")
        if input().lower() != "y":
            sys.exit(1)
        warn("QG will be disabled")
        return

    ok("Ollama found")

    # Try to start Ollama if not running
    try:
        import requests
        r = requests.get(OLLAMA_URL + "/api/tags", timeout=3)
        ok("Ollama daemon is running")
    except Exception:
        print("  Ollama daemon not running - starting it...")
        subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        import time
        time.sleep(3)

    # Pull model if needed
    print(f"  Pulling {OLLAMA_MODEL} (first time only, ~1GB)...")
    try:
        subprocess.run(["ollama", "pull", OLLAMA_MODEL], check=True)
        ok(f"{OLLAMA_MODEL} ready")
    except subprocess.CalledProcessError as exc:
        err(f"ollama pull failed: {exc}")
        warn("QG will be disabled")


def main() -> None:
    header("Auris — Model Setup (Wav2Vec2 Emotion - No Compilation!)")
    print("  Pipeline:  PyAV → Whisper → Flan-T5 → Wav2Vec2 Emotion (8 classes) → Qwen QG")
    print("  [OK] No C++ compilation required!")
    print("  [OK] Works on Windows out of the box!")
    print("  [OK] 8 emotion classes: Anger, Calm, Disgust, Fear, Happy, Neutral, Sad, Surprised")

    check_python()

    TOTAL = 6

    step(1, TOTAL, "Python packages")
    check_packages()

    step(2, TOTAL, "PyAV (bundled FFmpeg)")
    try:
        import av
        _ = av.audio.resampler.AudioResampler(format="s16", layout="mono", rate=16000)
        ok(f"PyAV {av.__version__} ready")
    except Exception as exc:
        err(f"PyAV check failed: {exc}")
        sys.exit(1)

    step(3, TOTAL, f"faster-whisper model ({WHISPER_MODEL})")
    download_whisper()

    step(4, TOTAL, f"Flan-T5 model ({FLAN_MODEL})")
    download_flan()

    step(5, TOTAL, f"Wav2Vec2 Emotion model (8 classes)")
    download_emotion_model()

    step(6, TOTAL, f"Ollama + Qwen QG ({OLLAMA_MODEL})")
    setup_ollama_qg()

    print()
    print(LINE)
    print("  [OK] Setup complete!")
    print()
    print("  Start the server:")
    print("    uvicorn backend.main:app --reload --port 8000")
    print()
    print("  Test transcription + emotion:")
    print('    curl -X POST "http://localhost:8000/transcribe" -F "file=@audio.mp3"')
    print()
    print("  Emotion classes: Anger, Calm, Disgust, Fear, Happy, Neutral, Sad, Surprised")
    print(LINE)


if __name__ == "__main__":
    main()