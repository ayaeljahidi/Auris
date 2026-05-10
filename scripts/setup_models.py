"""Auris — Model Setup Script (CPU-only, no compilation, Dual Emotion Fusion)"""
import os
import sys
import subprocess
import platform
from pathlib import Path

# ── Constants ──────────────────────────────────────────────────────────────────
FLAN_MODEL    = os.environ.get("FLAN_MODEL", "google/flan-t5-base")
FLAN_DIR      = Path("models/flan-t5-base")
WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "base.en")

# Audio emotion model (Wav2Vec2 - no compilation)
EMOTION_MODEL = os.environ.get(
    "EMOTION_MODEL", "prithivMLmods/Speech-Emotion-Classification"
)

# Text emotion model (DistilRoBERTa - no compilation)
TEXT_EMOTION_MODEL = os.environ.get(
    "TEXT_EMOTION_MODEL", "j-hartmann/emotion-english-distilroberta-base"
)

OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:1.5b")
OLLAMA_URL   = "http://localhost:11434"

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
        "sentencepiece", "huggingface_hub", "librosa", "soundfile",
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


def download_audio_emotion_model() -> None:
    """Download Wav2Vec2 audio emotion model — no compilation needed."""
    try:
        from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor
    except ImportError:
        err("transformers not installed")
        sys.exit(1)

    print(f"  Downloading Wav2Vec2 audio emotion model: {EMOTION_MODEL}")
    print("  Emotions: Anger, Calm, Disgust, Fear, Happy, Neutral, Sad, Surprised")
    print("  (~378 MB, downloads once and caches locally)")

    try:
        model     = Wav2Vec2ForSequenceClassification.from_pretrained(EMOTION_MODEL)
        processor = Wav2Vec2FeatureExtractor.from_pretrained(EMOTION_MODEL)

        num_labels = model.config.num_labels if hasattr(model.config, "num_labels") else 8
        labels     = model.config.id2label   if hasattr(model.config, "id2label")   else None

        import torch
        import numpy as np
        dummy  = np.zeros(16000, dtype=np.float32)
        inputs = processor(dummy, sampling_rate=16000, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
            logits  = outputs.logits

        ok(f"Wav2Vec2 audio emotion model ready — {num_labels} classes")
        if labels:
            ok(f"  Labels: {list(labels.values())}")

    except Exception as exc:
        err(f"Audio emotion model download failed: {exc}")
        print("\n  [WARN] Audio emotion detection will be disabled. Continue anyway? (y/n)")
        if input().lower() != "y":
            sys.exit(1)


def download_text_emotion_model() -> None:
    """Download DistilRoBERTa text emotion model — no compilation needed."""
    try:
        from transformers import pipeline as hf_pipeline
    except ImportError:
        err("transformers not installed")
        sys.exit(1)

    print(f"  Downloading DistilRoBERTa text emotion model: {TEXT_EMOTION_MODEL}")
    print("  Emotions: anger, disgust, fear, joy, neutral, sadness, surprise")
    print("  (~300 MB, downloads once and caches locally)")

    try:
        pipe = hf_pipeline(
            "text-classification",
            model=TEXT_EMOTION_MODEL,
            top_k=None,
            device=-1,
        )

        # Quick validation with a test sentence
        result = pipe("I am very happy today!")
        scores = result[0] if isinstance(result[0], list) else result
        best   = max(scores, key=lambda x: x["score"])

        ok(f"DistilRoBERTa text emotion model ready — {len(scores)} classes")
        ok(f"  Validation: 'I am very happy today!' → {best['label']} ({best['score']:.1%})")

    except Exception as exc:
        err(f"Text emotion model download failed: {exc}")
        print("\n  [WARN] Text emotion detection will be disabled. Continue anyway? (y/n)")
        if input().lower() != "y":
            sys.exit(1)


def setup_ollama_qg() -> None:
    """Ollama setup."""
    print()
    print(f"  QG model : {OLLAMA_MODEL}")
    print(f"  Endpoint : {OLLAMA_URL}")
    print()

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

    try:
        import requests
        r = requests.get(OLLAMA_URL + "/api/tags", timeout=3)
        ok("Ollama daemon is running")
    except Exception:
        print("  Ollama daemon not running - starting it...")
        subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        import time
        time.sleep(3)

    print(f"  Pulling {OLLAMA_MODEL} (first time only, ~1 GB)...")
    try:
        subprocess.run(["ollama", "pull", OLLAMA_MODEL], check=True)
        ok(f"{OLLAMA_MODEL} ready")
    except subprocess.CalledProcessError as exc:
        err(f"ollama pull failed: {exc}")
        warn("QG will be disabled")


def main() -> None:
    header("Auris — Model Setup (Dual Emotion Fusion — No Compilation!)")
    print("  Pipeline:  PyAV → Whisper → Flan-T5 → [Wav2Vec2 + DistilRoBERTa] → Fusion → Qwen QG")
    print()
    print("  [OK] No C++ compilation required!")
    print("  [OK] Works on Windows out of the box!")
    print("  [OK] Audio emotion  (8 classes): Anger, Calm, Disgust, Fear, Happy, Neutral, Sad, Surprised")
    print("  [OK] Text  emotion  (7 classes): Anger, Disgust, Fear, Joy, Neutral, Sadness, Surprise")
    print("  [OK] Fusion layer merges both signals for higher accuracy")

    check_python()

    TOTAL = 7

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

    step(5, TOTAL, "Wav2Vec2 Audio Emotion model (8 classes)")
    download_audio_emotion_model()

    step(6, TOTAL, "DistilRoBERTa Text Emotion model (7 classes)")
    download_text_emotion_model()

    step(7, TOTAL, f"Ollama + Qwen QG ({OLLAMA_MODEL})")
    setup_ollama_qg()

    print()
    print(LINE)
    print("  [OK] Setup complete!")
    print()
    print("  Start the server:")
    print("    uvicorn backend.main:app --reload --port 8000")
    print()
    print("  Test transcription + fused emotion:")
    print('    curl -X POST "http://localhost:8000/transcribe" -F "file=@audio.mp3"')
    print()
    print("  Disable text emotion (use audio only):")
    print("    TEXT_EMOTION_ENABLED=false uvicorn backend.main:app --reload --port 8000")
    print()
    print("  Custom fusion weights (e.g. trust audio more):")
    print("    EMOTION_AUDIO_WEIGHT=0.6 EMOTION_TEXT_WEIGHT=0.4 uvicorn backend.main:app --reload --port 8000")
    print(LINE)


if __name__ == "__main__":
    main()
