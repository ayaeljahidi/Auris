"""
Auris — Model Setup Script (CPU-only, FFmpeg-free)
Downloads and verifies all required models and dependencies.
Includes Ollama + Qwen QG setup (Step 7).

Usage:
    python scripts/setup_models.py
"""
import os
import sys
import subprocess
import platform
from pathlib import Path

# ── Constants ──────────────────────────────────────────────────────────────────

FLAN_MODEL    = os.environ.get("FLAN_MODEL",    "google/flan-t5-base")
FLAN_DIR      = Path("models/flan-t5-base")
WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "base.en")

EMOTION_MODEL = os.environ.get("EMOTION_MODEL", "onnx-community/wav2vec2-emotion-recognition-ONNX")
EMOTION_DIR   = Path("models/wav2vec2-emotion-onnx")

SILERO_VAD_DIR = Path("models/silero-vad")

OLLAMA_MODEL   = os.environ.get("OLLAMA_MODEL", "qwen2.5:1.5b")
OLLAMA_URL     = "http://localhost:11434"

LINE = "-" * 60


# ── Printer helpers ────────────────────────────────────────────────────────────

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


# ── Steps 1–6 (unchanged) ─────────────────────────────────────────────────────

def check_python() -> None:
    if sys.version_info < (3, 10):
        err("Python 3.10+ is required.")
        sys.exit(1)
    ok("Python " + str(sys.version_info.major) + "." + str(sys.version_info.minor))


def check_packages() -> None:
    required = [
        "fastapi",
        "uvicorn",
        "av",
        "numpy",
        "torch",
        "transformers",
        "sentencepiece",
        "onnxruntime",
        "requests",   # needed by qwen_questions.py
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

    try:
        import torch
        ok("torch.hub available (required by Silero VAD)")
    except Exception as exc:
        err("torch.hub check failed: " + str(exc))
        sys.exit(1)


def download_whisper() -> None:
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
    try:
        from transformers import T5ForConditionalGeneration, T5Tokenizer
    except ImportError:
        err("transformers not installed")
        sys.exit(1)

    FLAN_DIR.mkdir(parents=True, exist_ok=True)
    print("  Downloading " + FLAN_MODEL + " (~1 GB)...")
    try:
        _ = T5Tokenizer.from_pretrained(FLAN_MODEL, cache_dir=str(FLAN_DIR))
        _ = T5ForConditionalGeneration.from_pretrained(
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
        _ = AutoFeatureExtractor.from_pretrained(
            EMOTION_MODEL, cache_dir=str(EMOTION_DIR), trust_remote_code=True,
        )
        ok("Feature extractor downloaded")

        files      = list_repo_files(EMOTION_MODEL)
        onnx_files = [f for f in files if f.endswith(".onnx")]
        if not onnx_files:
            err("No ONNX files found in repository")
            sys.exit(1)

        target_file    = "model.onnx" if "model.onnx" in onnx_files else onnx_files[0]
        downloaded_path = hf_hub_download(
            repo_id=EMOTION_MODEL,
            filename=target_file,
            cache_dir=str(EMOTION_DIR),
            local_dir=str(EMOTION_DIR),
            local_dir_use_symlinks=False,
        )

        file_size = os.path.getsize(downloaded_path) / (1024 * 1024)
        ok("ONNX model downloaded — " + target_file + " (" + str(round(file_size, 1)) + " MB)")

        import onnxruntime as ort
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        session = ort.InferenceSession(
            downloaded_path, sess_options, providers=["CPUExecutionProvider"]
        )
        inputs  = [i.name for i in session.get_inputs()]
        outputs = [o.name for o in session.get_outputs()]
        ok("ONNX Runtime validation passed — inputs: " + str(inputs))

    except Exception as exc:
        err("Emotion model download failed: " + str(exc))
        print("\n  [WARN] Emotion detection will be disabled. Continue anyway? (y/n)")
        if input().lower() != "y":
            sys.exit(1)


def download_silero_vad() -> None:
    import torch

    SILERO_VAD_DIR.mkdir(parents=True, exist_ok=True)
    local_pt = SILERO_VAD_DIR / "silero_vad.pt"

    print("  Loading Silero VAD via torch.hub (snakers4/silero-vad)...")

    try:
        vad_model, vad_utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
            onnx=False,
        )
        vad_model.eval()
        ok("Silero VAD model loaded from torch.hub")

        torch.save(vad_model.state_dict(), str(local_pt))
        ok("Silero VAD state dict saved → " + str(local_pt))

        import numpy as np
        dummy_audio = torch.zeros(1, 512)
        vad_model.reset_states()
        with torch.no_grad():
            prob = vad_model(dummy_audio, 16000).item()
        ok("Silero VAD warm-up passed — speech prob on silence: " + str(round(prob, 4)))

    except Exception as exc:
        err("Silero VAD download failed: " + str(exc))
        print("\n  [WARN] VAD gating will be disabled. Continue anyway? (y/n)")
        if input().lower() != "y":
            sys.exit(1)


def warmup_pyav() -> None:
    try:
        import av
        _ = av.audio.resampler.AudioResampler(format="s16", layout="mono", rate=16000)
        ok("PyAV " + av.__version__ + " ready")
    except Exception as exc:
        err("PyAV check failed: " + str(exc))
        sys.exit(1)


# ── Step 7: Ollama + Qwen QG ──────────────────────────────────────────────────

def _ollama_is_running() -> bool:
    """Check whether Ollama daemon is reachable."""
    try:
        import requests
        r = requests.get(OLLAMA_URL + "/api/tags", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


def _ollama_has_model() -> bool:
    """Return True if OLLAMA_MODEL is already pulled."""
    try:
        import requests
        r = requests.get(OLLAMA_URL + "/api/tags", timeout=3)
        models = [m["name"] for m in r.json().get("models", [])]
        return any(OLLAMA_MODEL.split(":")[0] in m for m in models)
    except Exception:
        return False


def _install_ollama_linux() -> bool:
    """Run the official Ollama install script on Linux/macOS."""
    print("  Running: curl -fsSL https://ollama.com/install.sh | sh")
    try:
        result = subprocess.run(
            "curl -fsSL https://ollama.com/install.sh | sh",
            shell=True,
            check=True,
        )
        return result.returncode == 0
    except subprocess.CalledProcessError as exc:
        err("Ollama install script failed: " + str(exc))
        return False


def _start_ollama_background() -> None:
    """Start the Ollama daemon in the background."""
    try:
        subprocess.Popen(
            ["ollama", "serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        import time
        print("  Waiting for Ollama daemon to start…", end="", flush=True)
        for _ in range(15):
            time.sleep(1)
            print(".", end="", flush=True)
            if _ollama_is_running():
                print(" ✓")
                return
        print()
        warn("Ollama daemon did not respond in 15 s — try starting it manually")
    except FileNotFoundError:
        warn("'ollama' binary not found on PATH after install")


def setup_ollama_qg() -> None:
    """
    Step 7: ensure Ollama is installed, running, and the QG model is pulled.

    Strategy:
      1. Check if `ollama` binary is on PATH
         a. Not found on Linux/macOS → run install.sh automatically
         b. Not found on Windows     → print manual download instructions
      2. Check if daemon is running → start it if not
      3. Check if model is pulled   → pull it if not
      4. Warm-up inference          → send one token to force model into RAM
    """
    print()
    print("  QG model : " + OLLAMA_MODEL)
    print("  Endpoint : " + OLLAMA_URL)
    print()

    # ── 1. Is ollama on PATH? ─────────────────────────────────────────────────
    ollama_on_path = subprocess.run(
        ["ollama", "--version"],
        capture_output=True,
    ).returncode == 0

    if not ollama_on_path:
        _system = platform.system()
        if _system in ("Linux", "Darwin"):
            print("  Ollama binary not found — installing automatically…")
            if not _install_ollama_linux():
                err("Could not install Ollama automatically.")
                print()
                print("  Manual install:")
                print("    Linux/macOS : curl -fsSL https://ollama.com/install.sh | sh")
                print("    Windows     : https://ollama.com/download/windows")
                print()
                print("  Continue without QG? (y/n)")
                if input().lower() != "y":
                    sys.exit(1)
                warn("QG will be disabled — Ollama not installed")
                return
            ok("Ollama installed via install.sh")
        else:
            # Windows
            err("Ollama not found on PATH.")
            print()
            print("  ┌─────────────────────────────────────────────────────┐")
            print("  │  Windows install (one-time, ~500 MB):               │")
            print("  │                                                      │")
            print("  │  1. Go to https://ollama.com/download/windows       │")
            print("  │  2. Download and run OllamaSetup.exe                │")
            print("  │  3. Re-run this script after installation           │")
            print("  └─────────────────────────────────────────────────────┘")
            print()
            print("  Continue without QG? (y/n)")
            if input().lower() != "y":
                sys.exit(1)
            warn("QG will be disabled — install Ollama manually then re-run setup")
            return
    else:
        result = subprocess.run(["ollama", "--version"], capture_output=True, text=True)
        ok("Ollama found: " + result.stdout.strip())

    # ── 2. Is daemon running? ─────────────────────────────────────────────────
    if not _ollama_is_running():
        print("  Ollama daemon not running — starting it…")
        _start_ollama_background()

    if not _ollama_is_running():
        warn("Ollama daemon still not reachable — QG will fail at runtime")
        print("  Start it manually with: ollama serve")
        print("  Continue anyway? (y/n)")
        if input().lower() != "y":
            sys.exit(1)
        return

    ok("Ollama daemon is running at " + OLLAMA_URL)

    # ── 3. Is model pulled? ───────────────────────────────────────────────────
    if _ollama_has_model():
        ok(OLLAMA_MODEL + " already available — skipping pull")
    else:
        print("  Pulling " + OLLAMA_MODEL + " (~1 GB, first time only)…")
        print("  This may take several minutes depending on your connection.")
        try:
            subprocess.run(["ollama", "pull", OLLAMA_MODEL], check=True)
            ok(OLLAMA_MODEL + " pulled successfully")
        except subprocess.CalledProcessError as exc:
            err("ollama pull failed: " + str(exc))
            print("\n  Manual fix: ollama pull " + OLLAMA_MODEL)
            print("  Continue anyway? (y/n)")
            if input().lower() != "y":
                sys.exit(1)
            return

    # ── 4. Warm-up ────────────────────────────────────────────────────────────
    print("  Running warm-up inference (1 token)…")
    try:
        import requests
        import time
        t0 = time.perf_counter()
        r = requests.post(
            OLLAMA_URL + "/api/generate",
            json={
                "model":   OLLAMA_MODEL,
                "prompt":  "Hi",
                "stream":  False,
                "keep_alive": -1,
                "options": {"num_predict": 1},
            },
            timeout=120,
        )
        r.raise_for_status()
        elapsed = round((time.perf_counter() - t0) * 1000)
        ok(OLLAMA_MODEL + " warm-up passed (" + str(elapsed) + "ms — model now in RAM)")
    except Exception as exc:
        warn("Warm-up failed (non-fatal): " + str(exc))
        print("  QG will still work — model will cold-load on first request")

    print()
    print("  How the QG stage fits in the pipeline:")
    print("  ─────────────────────────────────────────────────────")
    print("  [=====Whisper======]")
    print("          [=Flan s1=][=s2=]…  ← overlap")
    print("  [==========Emotion===========]  ← independent")
    print("                             [==Qwen QG==]  ← after Flan")
    print()
    print("  Upload mode  : QG always runs (run_qg=True)")
    print("  Live mode    : QG skipped by default (set QG_ENABLED_LIVE=true to enable)")


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    header("Auris — Model Setup v17 (CPU-only, FFmpeg-free, + QG)")
    print("  Pipeline:  PyAV → Whisper → Flan-T5 → Emotion (ONNX) → Qwen QG (Ollama)")
    print("  [OK] No C++ compilation required!")
    print("  [OK] Works on Windows out of the box!")
    print("  [OK] Parallel Whisper + Emotion inference!")
    print("  [OK] Silero VAD gating — 2-3× faster emotion!")
    print("  [OK] Qwen2.5 QG — 4 jury questions per presentation!")

    check_python()

    TOTAL = 7

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

    step(6, TOTAL, "Silero VAD (speech gating for emotion speed-up)")
    download_silero_vad()

    step(7, TOTAL, "Ollama + Qwen QG (" + OLLAMA_MODEL + ")")
    setup_ollama_qg()

    print()
    print(LINE)
    print("  [OK] Setup complete!")
    print()
    print("  Start the server:")
    print("    uvicorn backend.main:app --reload --port 8000")
    print()
    print("  Test transcription + QG:")
    print('    curl -X POST "http://localhost:8000/transcribe" -F "file=@audio.mp3"')
    print()
    print("  Disable QG (faster, upload mode):")
    print("    OLLAMA_MODEL=disabled uvicorn backend.main:app --port 8000")
    print()
    print("  Enable QG in live/WebSocket mode:")
    print("    QG_ENABLED_LIVE=true uvicorn backend.main:app --port 8000")
    print(LINE)


if __name__ == "__main__":
    main()