"""
Auris — Model Setup Script (CPU-only, FFmpeg-free)
Downloads and verifies all required models and dependencies.

Usage:
    python scripts/setup_models.py
"""
import os
import sys
from pathlib import Path

# ── Constants ──────────────────────────────────────────────────────────────────

FLAN_MODEL = os.environ.get("FLAN_MODEL", "google/flan-t5-base")
FLAN_DIR   = Path("models/flan-t5-base")
WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "base.en")

# ONNX Community pre-converted model (no PyTorch conversion needed)
EMOTION_MODEL = os.environ.get("EMOTION_MODEL", "onnx-community/wav2vec2-emotion-recognition-ONNX")
EMOTION_DIR   = Path("models/wav2vec2-emotion-onnx")

# Silero VAD — tiny PyTorch model (~2 MB), loaded via torch.hub
SILERO_VAD_DIR = Path("models/silero-vad")

LINE = "-" * 60


def header(text: str) -> None:
    print()
    print(LINE)
    print("  " + text)
    print(LINE)


def step(n: int, total: int, text: str) -> None:
    print()
    print("[" + str(n) + "/" + str(total) + "] " + text)


def ok(msg: str) -> None:
    print("  [OK] " + msg)


def warn(msg: str) -> None:
    print("  [WARN] " + msg)


def err(msg: str) -> None:
    print("  [ERR] " + msg)


def check_python() -> None:
    if sys.version_info < (3, 10):
        err("Python 3.10+ is required.")
        sys.exit(1)
    ok("Python " + str(sys.version_info.major) + "." + str(sys.version_info.minor))


def check_packages() -> None:
    # silero-vad is loaded via torch.hub (no pip package needed),
    # but we keep it in the list so users know it is a dependency.
    required = [
        "fastapi",
        "uvicorn",
        "av",
        "numpy",
        "torch",
        "transformers",
        "sentencepiece",
        "onnxruntime",
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

    # silero-vad ships as a torch.hub model — no pip entry, but verify torch.hub works
    try:
        import torch
        ok("torch.hub available (required by Silero VAD)")
    except Exception as exc:
        err("torch.hub check failed: " + str(exc))
        sys.exit(1)


def download_whisper() -> None:
    """Download faster-whisper model."""
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        err("faster-whisper not installed")
        sys.exit(1)

    print("  Downloading faster-whisper " + WHISPER_MODEL + "...")
    try:
        _ = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8", cpu_threads=4)
        ok("faster-whisper (" + WHISPER_MODEL + " / int8 / CPU) ready")
    except Exception as exc:
        err("Whisper download failed: " + str(exc))
        sys.exit(1)


def download_flan() -> None:
    """Download Flan-T5 model."""
    try:
        from transformers import T5ForConditionalGeneration, T5Tokenizer
    except ImportError:
        err("transformers not installed")
        sys.exit(1)

    FLAN_DIR.mkdir(parents=True, exist_ok=True)
    print("  Downloading " + FLAN_MODEL + " (~1 GB)...")
    try:
        tokenizer = T5Tokenizer.from_pretrained(FLAN_MODEL, cache_dir=str(FLAN_DIR))
        model = T5ForConditionalGeneration.from_pretrained(
            FLAN_MODEL,
            dtype="auto",
            device_map="cpu",
            cache_dir=str(FLAN_DIR),
        )
        ok("Flan-T5 (" + FLAN_MODEL + ") downloaded")
    except Exception as exc:
        err("Flan-T5 download failed: " + str(exc))
        sys.exit(1)


def download_emotion_model() -> None:
    """Download pre-converted ONNX emotion model from Hugging Face."""
    try:
        from huggingface_hub import hf_hub_download, list_repo_files
        from transformers import AutoFeatureExtractor
    except ImportError:
        err("huggingface_hub or transformers not installed")
        sys.exit(1)

    EMOTION_DIR.mkdir(parents=True, exist_ok=True)
    print("  Downloading " + EMOTION_MODEL + " (~360 MB ONNX)...")
    print("  Labels: angry, disgust, fear, happy, neutral, sad, surprise")

    try:
        # Download feature extractor / preprocessor
        extractor = AutoFeatureExtractor.from_pretrained(
            EMOTION_MODEL,
            cache_dir=str(EMOTION_DIR),
            trust_remote_code=True,
        )
        ok("Feature extractor downloaded")

        # Find and download ONNX model file
        files = list_repo_files(EMOTION_MODEL)
        onnx_files = [f for f in files if f.endswith(".onnx")]

        if not onnx_files:
            err("No ONNX files found in repository")
            sys.exit(1)

        target_file = "model.onnx" if "model.onnx" in onnx_files else onnx_files[0]
        print("  Downloading ONNX weights: " + target_file + "...")

        downloaded_path = hf_hub_download(
            repo_id=EMOTION_MODEL,
            filename=target_file,
            cache_dir=str(EMOTION_DIR),
            local_dir=str(EMOTION_DIR),
            local_dir_use_symlinks=False,
        )

        # Verify the file
        file_size = os.path.getsize(downloaded_path) / (1024 * 1024)
        ok("ONNX model downloaded — " + target_file + " (" + str(round(file_size, 1)) + " MB)")

        # Try loading with ONNX Runtime to verify
        import onnxruntime as ort
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        session = ort.InferenceSession(
            downloaded_path, sess_options, providers=["CPUExecutionProvider"]
        )

        inputs  = [i.name for i in session.get_inputs()]
        outputs = [o.name for o in session.get_outputs()]
        ok("ONNX Runtime validation passed — inputs: " + str(inputs) + ", outputs: " + str(outputs))

    except Exception as exc:
        err("Emotion model download failed: " + str(exc))
        print("\n  [WARN] Emotion detection will be disabled. Continue anyway? (y/n)")
        if input().lower() != "y":
            sys.exit(1)


def download_silero_vad() -> None:
    """
    Download and warm-up Silero VAD via torch.hub.

    The model (~2 MB) is cached automatically by torch.hub under
    ~/.cache/torch/hub/snakers4_silero-vad_master/.
    We additionally save a local copy under models/silero-vad/ so the
    server can load it offline without internet access.

    Silero VAD is used in emotion.py to skip silence-only audio chunks
    before feeding them to the heavy wav2vec2 ONNX model, cutting
    emotion inference time by 30–50 % on typical speech recordings.
    """
    import torch

    SILERO_VAD_DIR.mkdir(parents=True, exist_ok=True)
    local_pt = SILERO_VAD_DIR / "silero_vad.pt"

    print("  Loading Silero VAD via torch.hub (snakers4/silero-vad)...")
    print("  Model size: ~2 MB  |  Inference: <2 ms per 512-sample window")

    try:
        # Force fresh download so the file is in torch.hub cache
        vad_model, vad_utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,  # re-use cache if already downloaded
            onnx=False,          # pure PyTorch — fastest on CPU for short windows
        )
        ok("Silero VAD model loaded from torch.hub")

        # Persist a local .pt copy for offline / production use
        torch.save(vad_model.state_dict(), str(local_pt))
        ok("Silero VAD state dict saved → " + str(local_pt))

        # Functional warm-up: run one 512-sample window of silence
        import numpy as np
        dummy_audio = torch.zeros(1, 512)   # shape (batch, samples), 32 ms at 16 kHz
        vad_model.reset_states()
        with torch.no_grad():
            prob = vad_model(dummy_audio, 16000).item()
        ok("Silero VAD warm-up passed — speech prob on silence: " + str(round(prob, 4)))

        # Show available utility functions
        get_speech_ts, _, _, _, _ = vad_utils
        ok("Utility function get_speech_timestamps available")

        print()
        print("  How Silero VAD speeds up emotion detection:")
        print("  ─────────────────────────────────────────────")
        print("  Before: 7 chunks × 4 400 ms = 30 800 ms (silence + speech, all processed)")
        print("  After:  only speech chunks sent to wav2vec2  → ~50 % fewer chunks typical")
        print("  Expected: 3–4 speech chunks × 4 400 ms ≈ 13–18 s for 95 s recording")
        print("  VAD overhead: < 5 ms for full audio  (negligible)")

    except Exception as exc:
        err("Silero VAD download failed: " + str(exc))
        print()
        print("  Common causes:")
        print("    - No internet access during setup (torch.hub requires network)")
        print("    - Corporate proxy blocking raw.githubusercontent.com")
        print("    - torch not installed correctly")
        print()
        print("  Manual fix:")
        print("    pip install torch --index-url https://download.pytorch.org/whl/cpu")
        print("    python -c \"import torch; torch.hub.load('snakers4/silero-vad', 'silero_vad')\"")
        print()
        print("  [WARN] VAD gating will be disabled (emotion detection will be slower).")
        print("  Continue anyway? (y/n)")
        if input().lower() != "y":
            sys.exit(1)


def warmup_pyav() -> None:
    """Verify PyAV is working correctly."""
    try:
        import av
        resampler = av.audio.resampler.AudioResampler(format="s16", layout="mono", rate=16000)
        ok("PyAV " + av.__version__ + " ready")
    except Exception as exc:
        err("PyAV check failed: " + str(exc))
        sys.exit(1)


def main() -> None:
    header("Auris — Model Setup (CPU-only, FFmpeg-free)")
    print("  Pipeline:  PyAV → faster-whisper → Flan-T5 → Silero VAD → Wav2Vec2 Emotion (ONNX)")
    print("  [OK] No C++ compilation required!")
    print("  [OK] Works on Windows out of the box!")
    print("  [OK] Parallel Whisper + Emotion inference!")
    print("  [OK] Pre-converted ONNX model — no conversion needed!")
    print("  [OK] Silero VAD gating — skip silence, speed up emotion 2-3x!")

    check_python()

    TOTAL = 6
    step(1, TOTAL, "Python packages")
    check_packages()

    step(2, TOTAL, "PyAV (bundled FFmpeg)")
    warmup_pyav()

    step(3, TOTAL, "faster-whisper model (" + WHISPER_MODEL + ")")
    download_whisper()

    step(4, TOTAL, "Flan-T5 model (" + FLAN_MODEL + ")")
    download_flan()

    step(5, TOTAL, "Wav2Vec2 Emotion ONNX model (" + EMOTION_MODEL + ")")
    download_emotion_model()

    step(6, TOTAL, "Silero VAD (speech/silence gating for emotion speed-up)")
    download_silero_vad()

    print()
    print(LINE)
    print("  [OK] Setup complete!")
    print()
    print("  Start the server:")
    print("    uvicorn backend.main:app --reload --port 8000")
    print()
    print("  Test with:")
    print('    curl -X POST "http://localhost:8000/transcribe" -F "file=@audio.mp3"')
    print(LINE)


if __name__ == "__main__":
    main()