"""Auris — FastAPI application entry point (CPU-optimized, VAD removed, FFmpeg-free)"""
import asyncio
import logging
import time
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from .audio import extract_audio, read_wav, pcm_to_wav, audio_duration_seconds
from .models import load_whisper, load_flan, health_status
from .transcribe import (
    transcribe_whisper,
    transcribe_whisper_with_correction,
)
from . import config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)-8s]  %(name)s — %(message)s",
)
log = logging.getLogger("auris")

# ── App ────────────────────────────────────────────────────────────────────────

app = FastAPI(title="Auris API", version="10.0", docs_url="/api/docs")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Startup ────────────────────────────────────────────────────────────────────

@app.on_event("startup")
async def on_startup() -> None:
    """Pre-load all models in background threads so the first request is fast."""
    loop = asyncio.get_event_loop()
    log.info("Pre-loading models (CPU-optimized, VAD removed, FFmpeg-free)…")
    for loader, name in [
        (load_whisper, "faster-whisper"),
        (load_flan,    "Flan-T5"),
    ]:
        try:
            await loop.run_in_executor(None, loader)
        except Exception as exc:
            log.warning("%s warmup skipped: %s", name, exc)
    log.info("✓ All models ready")


# ── Health ─────────────────────────────────────────────────────────────────────

@app.get("/health", tags=["system"])
async def health() -> dict:
    return {"status": "ok", "version": "10.0", **health_status()}


# ── Transcribe (file upload) ───────────────────────────────────────────────────

@app.post("/transcribe", tags=["transcription"])
async def transcribe(file: UploadFile = File(...)):
    """
    Full pipeline:  upload → PyAV → Whisper + Flan-T5 (with critique)

    VAD and FFmpeg subprocess removed. PyAV handles extraction in-memory.
    Total latency = t(PyAV) + t(Whisper + Flan-T5)
    """
    t_start   = time.perf_counter()
    raw_bytes = await file.read()
    log.info("Received '%s'  (%d KB)", file.filename, len(raw_bytes) // 1024)

    loop = asyncio.get_event_loop()

    # 1 · PyAV extraction (in-memory, no temp files) ----------------------------
    t0 = time.perf_counter()
    try:
        wav_bytes = await loop.run_in_executor(None, extract_audio, raw_bytes)
    except Exception as exc:
        return JSONResponse(
            {"status": "error", "message": f"Audio extraction failed: {exc}"},
            status_code=422,
        )
    pcm, sr  = read_wav(wav_bytes)
    duration = audio_duration_seconds(pcm, sr)
    t_extract = _ms(t0)

    # 2 · Whisper + Flan-T5 (batched segment correction) -------------------------
    t1 = time.perf_counter()
    try:
        whisper_result, correction = await loop.run_in_executor(
            None, transcribe_whisper_with_correction, wav_bytes
        )
    except Exception as exc:
        return JSONResponse(
            {"status": "error", "message": f"Transcription failed: {exc}"},
            status_code=500,
        )
    t_whisper = _ms(t1)

    t_total = _ms(t_start)

    log.info(
        "Done — extract:%dms  whisper+flan:%dms  total:%dms  "
        "critique(corrected:%d/kept:%d)",
        t_extract, t_whisper, t_total,
        correction.get("critique_stats", {}).get("corrected", 0),
        correction.get("critique_stats", {}).get("kept", 0),
    )

    return {
        "status":       "ok",
        "filename":     file.filename,
        "duration_sec": round(duration, 2),
        "whisper":      whisper_result,
        "correction":   correction,
        "timing": {
            "extract_ms":  t_extract,
            "whisper_ms":  t_whisper,
            "total_ms":    t_total,
        },
    }


# ── Live WebSocket ─────────────────────────────────────────────────────────────

@app.websocket("/ws/live")
async def ws_live(ws: WebSocket) -> None:
    """
    Live recording endpoint with configurable idle timeout.
    VAD removed — raw audio goes straight to Whisper.
    Flan-T5 is SKIPPED by default on live (config.FLAN_ENABLED_LIVE=false).

    Protocol:
      client  →  raw 16-bit mono PCM chunks (16 kHz)
      client  →  b"__END__"           (signals end of recording)
      server  →  {"type":"partial",   "text": "Recording… speak now"}
      server  →  {"type":"status",    "msg": …}
      server  →  {"type":"final",     "whisper": …, "correction": …}
      server  →  {"type":"error",     "msg": …}
    """
    await ws.accept()
    log.info("Live WebSocket connected")

    pcm_buffer = bytearray()
    loop       = asyncio.get_event_loop()

    try:
        while True:
            try:
                data = await asyncio.wait_for(
                    ws.receive_bytes(),
                    timeout=config.WS_LIVE_TIMEOUT,
                )
            except asyncio.TimeoutError:
                log.warning("Live WebSocket timed out after %ds",
                            config.WS_LIVE_TIMEOUT)
                try:
                    await ws.send_json({"type": "error", "msg": "Recording timeout"})
                except Exception:
                    pass
                await ws.close()
                return

            # ── End-of-recording sentinel ──────────────────────────────────────
            if data == b"__END__":
                await ws.send_json({"type": "status", "msg": "Running Whisper…"})
                full_wav = pcm_to_wav(bytes(pcm_buffer), 16_000)

                if len(pcm_buffer) > 3_200:   # > 0.1 s of audio
                    try:
                        if config.FLAN_ENABLED_LIVE:
                            whisper_r, correction = await loop.run_in_executor(
                                None, transcribe_whisper_with_correction, full_wav
                            )
                        else:
                            whisper_r = await loop.run_in_executor(
                                None, transcribe_whisper, full_wav
                            )
                            correction = {
                                "corrected":  whisper_r["text"],
                                "enabled":    False,
                                "model":      None,
                                "latency_ms": 0,
                                "critique_stats": {"corrected": 0, "kept": 0, "total": 0},
                            }
                    except Exception as exc:
                        log.error("Live final error: %s", exc)
                        whisper_r  = {"text": "", "word_count": 0, "segments": []}
                        correction = {
                            "corrected":  "",
                            "enabled":    False,
                            "model":      None,
                            "latency_ms": 0,
                            "critique_stats": {"corrected": 0, "kept": 0, "total": 0},
                        }
                else:
                    whisper_r  = {"text": "", "word_count": 0, "segments": []}
                    correction = {
                        "corrected":  "",
                        "enabled":    False,
                        "model":      None,
                        "latency_ms": 0,
                        "critique_stats": {"corrected": 0, "kept": 0, "total": 0},
                    }

                await ws.send_json({
                    "type":         "final",
                    "whisper":      whisper_r,
                    "correction":   correction,
                })
                break

            # ── Streaming chunk received ───────────────────────────────────────
            pcm_buffer.extend(data)
            await ws.send_json({
                "type": "partial",
                "text": "Recording… speak now",
            })

    except WebSocketDisconnect:
        log.info("Live WebSocket disconnected")
    except Exception as exc:
        log.error("WebSocket error: %s", exc)
        try:
            await ws.send_json({"type": "error", "msg": str(exc)})
        except Exception:
            pass


# ── Static frontend ────────────────────────────────────────────────────────────

_frontend = Path(__file__).parent.parent / "frontend"
if _frontend.exists():
    app.mount("/", StaticFiles(directory=str(_frontend), html=True), name="static")


# ── Helpers ────────────────────────────────────────────────────────────────────

def _ms(t: float) -> int:
    """Elapsed milliseconds since perf_counter snapshot t."""
    return round((time.perf_counter() - t) * 1000)