"""
Vosper — Model Setup Script
Downloads and verifies all required models and dependencies.

Usage:
    python scripts/setup_models.py
"""
import io
import os
import shutil
import subprocess
import sys
import tempfile
import urllib.request
import wave
import zipfile
from pathlib import Path

# ── Constants ──────────────────────────────────────────────────────────────────

VOSK_URL  = "https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip"
VOSK_DIR  = Path("models/vosk/small")

SILERO_URL     = "https://github.com/snakers4/silero-vad/raw/v4.0/files/silero_vad.onnx"
SILERO_DIR  = Path("models/silero_vad")
SILERO_ONNX = SILERO_DIR / "silero_vad.onnx"

REQUIRED_PACKAGES = {
    "fastapi":        "fastapi",
    "uvicorn":        "uvicorn",
    "vosk":           "vosk",
    "faster_whisper": "faster-whisper",
    "torch":          "torch",
    "numpy":          "numpy",
    "onnxruntime":    "onnxruntime",
    "requests":       "requests",
}

LINE = "─" * 60


# ── Helpers ────────────────────────────────────────────────────────────────────

def header(text: str) -> None:
    print(f"\n{LINE}")
    print(f"  {text}")
    print(LINE)


def step(n: int, total: int, text: str) -> None:
    print(f"\n[{n}/{total}] {text}")


def ok(msg: str)   -> None: print(f"  ✓  {msg}")
def warn(msg: str) -> None: print(f"  !  {msg}")
def err(msg: str)  -> None: print(f"  ✗  {msg}")


def progress(count: int, block: int, total: int) -> None:
    if total > 0:
        pct  = min(int(count * block * 100 / total), 100)
        bar  = "█" * (pct // 2)
        print(f"\r  [{bar:<50}] {pct}%", end="", flush=True)


def check_python() -> None:
    if sys.version_info < (3, 9):
        err("Python 3.9+ is required.")
        sys.exit(1)
    ok(f"Python {sys.version_info.major}.{sys.version_info.minor}")


# ── Step implementations ───────────────────────────────────────────────────────

def check_ffmpeg() -> None:
    try:
        r = subprocess.run(["ffmpeg", "-version"], capture_output=True, timeout=10)
        if r.returncode == 0:
            ok("FFmpeg found")
        else:
            raise FileNotFoundError
    except Exception:
        err("FFmpeg not found")
        print("     Install:  sudo apt install ffmpeg  /  brew install ffmpeg")
        sys.exit(1)


def check_packages() -> None:
    missing = []
    for import_name, pkg_name in REQUIRED_PACKAGES.items():
        try:
            __import__(import_name)
            ok(pkg_name)
        except ImportError:
            err(pkg_name)
            missing.append(pkg_name)
    if missing:
        print(f"\n  Run:  pip install -r requirements.txt")
        sys.exit(1)


def download_vosk() -> None:
    if VOSK_DIR.exists():
        ok(f"Already installed at {VOSK_DIR}")
        return

    print("  Downloading (~50 MB)…")
    urllib.request.urlretrieve(VOSK_URL, "vosk.zip", progress)
    print()

    print("  Extracting…")
    with zipfile.ZipFile("vosk.zip") as zf:
        zf.extractall(".")

    VOSK_DIR.parent.mkdir(parents=True, exist_ok=True)
    shutil.move("vosk-model-small-en-us-0.15", str(VOSK_DIR))
    Path("vosk.zip").unlink(missing_ok=True)
    ok(f"Installed at {VOSK_DIR}")


def download_vad() -> None:
    """Download Silero VAD ONNX — no NeMo or NVIDIA account needed."""
    SILERO_DIR.mkdir(parents=True, exist_ok=True)

    if SILERO_ONNX.exists():
        ok(f"Already installed at {SILERO_ONNX}")
        return

    print("  Downloading Silero VAD ONNX (~2 MB)…")
    try:
        urllib.request.urlretrieve(SILERO_URL, SILERO_ONNX, progress)
        print()
        size = SILERO_ONNX.stat().st_size / 1024 / 1024
        ok(f"Installed at {SILERO_ONNX} ({size:.1f} MB)")
    except Exception as exc:
        print()
        err(f"Download failed: {exc}")
        sys.exit(1)


def warmup_whisper() -> None:
    try:
        import numpy as np
        from faster_whisper import WhisperModel

        model = WhisperModel("base.en", device="cpu", compute_type="int8")

        silence = np.zeros(16_000, dtype=np.int16)
        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(16_000)
            w.writeframes(silence.tobytes())

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(buf.getvalue())
            tmp = f.name

        list(model.transcribe(tmp, beam_size=1)[0])
        Path(tmp).unlink(missing_ok=True)
        ok("faster-whisper (base.en / int8) ready")
    except Exception as exc:
        warn(f"Whisper warmup failed: {exc}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    header("Vosper — Model Setup")
    print("  Pipeline:  FFmpeg → Silero VAD → Vosk ∥ faster-whisper")

    check_python()

    TOTAL = 5
    step(1, TOTAL, "FFmpeg")
    check_ffmpeg()

    step(2, TOTAL, "Python packages")
    check_packages()

    step(3, TOTAL, "Vosk small-en model")
    download_vosk()

    step(4, TOTAL, "Silero VAD ONNX")
    download_vad()

    step(5, TOTAL, "faster-whisper warmup")
    warmup_whisper()

    print(f"\n{LINE}")
    print("  Setup complete!\n")
    print("  Start the server:")
    print("    uvicorn backend.main:app --reload --port 8000")
    print(LINE)


if __name__ == "__main__":
    main()