"""
Auris -- Model Setup Script (CPU-only, FFmpeg-free, VAD-free)
Downloads and verifies all required models and dependencies.

Usage:
    python scripts/setup_models.py
"""
import io
import os
import sys
import tempfile
import wave
from pathlib import Path

# -- Constants -----------------------------------------------------------------

FLAN_MODEL    = os.environ.get("FLAN_MODEL",    "google/flan-t5-base")
FLAN_DIR      = Path("models/flan-t5-base")
QGEN_MODEL    = os.environ.get("QGEN_MODEL",    "valhalla/t5-base-qg-hl")
QGEN_DIR      = Path("models/t5-base-qg-hl")
WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "base.en")

# Exact packages from requirements.txt
REQUIRED_PACKAGES = {
    "fastapi":        "fastapi>=0.115.0",
    "uvicorn":        "uvicorn[standard]>=0.32.0",
    "multipart":      "python-multipart>=0.0.17",
    "av":             "av>=13.1.0",
    "faster_whisper": "faster-whisper>=1.1.0",
    "torch":          "torch>=2.5.0",
    "transformers":   "transformers>=4.46.0",
    "sentencepiece":  "sentencepiece>=0.2.0",
    "numpy":          "numpy>=1.26.0",
    "onnxruntime":    "onnxruntime>=1.20.0",
}

LINE = "-" * 60


# -- Helpers -------------------------------------------------------------------

def header(text: str) -> None:
    print(f"\n{LINE}")
    print(f"  {text}")
    print(LINE)


def step(n: int, total: int, text: str) -> None:
    print(f"\n[{n}/{total}] {text}")


def ok(msg: str)   -> None: print(f"  OK  {msg}")
def warn(msg: str) -> None: print(f"  !   {msg}")
def err(msg: str)  -> None: print(f"  ERR {msg}")


def check_python() -> None:
    if sys.version_info < (3, 10):
        err("Python 3.10+ is required.")
        sys.exit(1)
    ok(f"Python {sys.version_info.major}.{sys.version_info.minor}")


# -- Step implementations ------------------------------------------------------


def check_packages() -> None:
    missing = []
    for import_name, pkg_spec in REQUIRED_PACKAGES.items():
        try:
            __import__(import_name)
            ok(pkg_spec)
        except ImportError:
            err(pkg_spec)
            missing.append(pkg_spec)
    if missing:
        print(f"\n  Missing packages. Install with:")
        print(f"    pip install -r requirements.txt")
        sys.exit(1)


def download_whisper() -> None:
    """Download faster-whisper model (auto-cached by CTranslate2)."""
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        err("faster-whisper not installed")
        sys.exit(1)

    cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
    model_name_safe = f"Systran--faster-whisper-{WHISPER_MODEL.replace('.', '-')}"
    if any(model_name_safe in str(p) for p in cache_dir.glob("*")):
        ok(f"faster-whisper ({WHISPER_MODEL}) already cached")
        return

    print(f"  Downloading faster-whisper {WHISPER_MODEL} (CPU-optimized)...")
    print("  This may take a few minutes on first run.")
    try:
        _ = WhisperModel(
            WHISPER_MODEL,
            device="cpu",
            compute_type="int8",
            cpu_threads=4,
        )
        ok(f"faster-whisper ({WHISPER_MODEL} / int8 / CPU) ready")
    except Exception as exc:
        err(f"Whisper download failed: {exc}")
        sys.exit(1)


def download_flan() -> None:
    """Download Flan-T5 model using transformers (CPU-friendly)."""
    try:
        from transformers import T5ForConditionalGeneration, T5Tokenizer
    except ImportError:
        err("transformers not installed -- cannot download Flan-T5")
        sys.exit(1)

    FLAN_DIR.mkdir(parents=True, exist_ok=True)

    cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
    model_name_safe = FLAN_MODEL.replace("/", "--")
    if any(model_name_safe in str(p) for p in cache_dir.glob("*")):
        ok(f"Flan-T5 already cached")
        return

    print(f"  Downloading {FLAN_MODEL} (CPU-optimized, ~1 GB)...")
    print("  This may take a few minutes on first run.")
    try:
        import torch
        tokenizer = T5Tokenizer.from_pretrained(FLAN_MODEL, cache_dir=str(FLAN_DIR))
        model = T5ForConditionalGeneration.from_pretrained(
            FLAN_MODEL,
            torch_dtype=torch.float32,
            device_map="cpu",
            cache_dir=str(FLAN_DIR),
        )
        model.eval()
        # Force full weight materialisation before returning
        dummy = tokenizer("warmup", return_tensors="pt")
        with torch.no_grad():
            model.generate(**dummy, max_new_tokens=1)
        ok(f"Flan-T5 ({FLAN_MODEL}) downloaded and ready")
    except Exception as exc:
        err(f"Flan-T5 download failed: {exc}")
        sys.exit(1)


def download_qgen() -> None:
    """Download T5-QG model using transformers (CPU-friendly)."""
    try:
        from transformers import T5ForConditionalGeneration, T5Tokenizer
    except ImportError:
        err("transformers not installed -- cannot download T5-QG")
        sys.exit(1)

    QGEN_DIR.mkdir(parents=True, exist_ok=True)

    cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
    model_name_safe = QGEN_MODEL.replace("/", "--")
    if any(model_name_safe in str(p) for p in cache_dir.glob("*")):
        ok(f"T5-QG already cached")
        return

    print(f"  Downloading {QGEN_MODEL} (CPU-optimized, ~220 MB)...")
    print("  This may take a few minutes on first run.")
    try:
        import torch
        tokenizer = T5Tokenizer.from_pretrained(QGEN_MODEL, cache_dir=str(QGEN_DIR))
        model = T5ForConditionalGeneration.from_pretrained(
            QGEN_MODEL,
            torch_dtype=torch.float32,
            device_map="cpu",
            cache_dir=str(QGEN_DIR),
        )
        model.eval()
        # Force full weight materialisation before returning
        dummy = tokenizer("warmup", return_tensors="pt")
        with torch.no_grad():
            model.generate(**dummy, max_new_tokens=1)
        ok(f"T5-QG ({QGEN_MODEL}) downloaded and ready")
    except Exception as exc:
        err(f"T5-QG download failed: {exc}")
        sys.exit(1)


def warmup_whisper() -> None:
    try:
        import numpy as np
        from faster_whisper import WhisperModel

        model = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8", cpu_threads=4)

        silence = np.zeros(16_000, dtype=np.int16)
        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(16_000)
            w.writeframes(silence.tobytes())

        audio = silence.astype(np.float32) / 32768.0
        list(model.transcribe(audio, beam_size=1)[0])
        ok(f"faster-whisper ({WHISPER_MODEL} / int8 / CPU) warmed up")
    except Exception as exc:
        warn(f"Whisper warmup failed: {exc}")


def warmup_flan() -> None:
    """Warm up Flan-T5 with a tiny inference to trigger any lazy downloads."""
    try:
        from transformers import T5ForConditionalGeneration, T5Tokenizer
        import torch

        tokenizer = T5Tokenizer.from_pretrained(FLAN_MODEL, cache_dir=str(FLAN_DIR))
        model = T5ForConditionalGeneration.from_pretrained(
            FLAN_MODEL,
            torch_dtype=torch.float32,
            device_map="cpu",
            cache_dir=str(FLAN_DIR),
        )
        model.eval()

        inputs = tokenizer("test", return_tensors="pt")
        with torch.no_grad():
            _ = model.generate(**inputs, max_new_tokens=4)
        ok(f"Flan-T5 ({FLAN_MODEL} / CPU / float32) warmed up")
    except Exception as exc:
        warn(f"Flan-T5 warmup failed: {exc}")


def warmup_qgen() -> None:
    """Warm up T5-QG with a tiny inference."""
    try:
        from transformers import T5ForConditionalGeneration, T5Tokenizer
        import torch

        tokenizer = T5Tokenizer.from_pretrained(QGEN_MODEL, cache_dir=str(QGEN_DIR))
        model = T5ForConditionalGeneration.from_pretrained(
            QGEN_MODEL,
            torch_dtype=torch.float32,
            device_map="cpu",
            cache_dir=str(QGEN_DIR),
        )
        model.eval()

        inputs = tokenizer("generate question: test context", return_tensors="pt")
        with torch.no_grad():
            _ = model.generate(**inputs, max_new_tokens=8)
        ok(f"T5-QG ({QGEN_MODEL} / CPU / float32) warmed up")
    except Exception as exc:
        warn(f"T5-QG warmup failed: {exc}")


def warmup_pyav() -> None:
    """Verify PyAV is working correctly."""
    try:
        import av
        resampler = av.audio.resampler.AudioResampler(
            format="s16", layout="mono", rate=16000
        )
        ok(f"PyAV {av.__version__} ready (FFmpeg bundled)")
    except Exception as exc:
        err(f"PyAV check failed: {exc}")
        sys.exit(1)


# -- Main ----------------------------------------------------------------------

def main() -> None:
    header("Auris -- Model Setup (CPU-only, FFmpeg-free)")
    print("  Pipeline:  PyAV -> faster-whisper -> Flan-T5 (critique) -> T5-QG (questions)")
    print("  No system FFmpeg required -- PyAV bundles its own codecs")
    print("  No VAD -- frontend noise gate handles pre-filtering")

    check_python()

    TOTAL = 5
    step(1, TOTAL, "Python packages")
    check_packages()

    step(2, TOTAL, "PyAV (bundled FFmpeg)")
    warmup_pyav()

    step(3, TOTAL, f"faster-whisper model ({WHISPER_MODEL})")
    download_whisper()

    step(4, TOTAL, f"Flan-T5 model ({FLAN_MODEL})")
    download_flan()

    step(5, TOTAL, f"T5-QG model ({QGEN_MODEL})")
    download_qgen()

    # Warmups run after all downloads, no step counter shown
    print(f"\n[warmup] faster-whisper")
    warmup_whisper()
    print(f"[warmup] Flan-T5")
    warmup_flan()
    print(f"[warmup] T5-QG")
    warmup_qgen()

    print(f"\n{LINE}")
    print("  Setup complete!\n")
    print("  Start the server:")
    print("    uvicorn app.main:app --reload --port 8000")
    print(LINE)


if __name__ == "__main__":
    main()