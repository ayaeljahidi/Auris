"""Auris -- FastAPI application (zero-copy, aggressively optimized)"""
import asyncio
import logging
import time
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from .audio import extract_audio_to_numpy, pcm_to_wav, audio_duration_from_array
from .models import load_whisper, load_flan, load_qgen, health_status
from .transcribe import transcribe_whisper, transcribe_whisper_with_correction
from .transcribe import _log_questions_terminal
from . import config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)-8s]  %(name)s -- %(message)s",
)
log = logging.getLogger("auris")

# -- App -----------------------------------------------------------------------

app = FastAPI(title="Auris API", version="11.1", docs_url="/api/docs")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -- Startup -------------------------------------------------------------------

@app.on_event("startup")
async def on_startup() -> None:
    """Pre-load all models in background threads."""
    log.info("Pre-loading models (zero-copy, CPU-optimized)...")
    loaders = [
        (load_whisper, "faster-whisper"),
        (load_flan,    "Flan-T5"),
        (load_qgen,    "T5-QG"),
    ]
    for loader, name in loaders:
        try:
            await asyncio.to_thread(loader)
        except Exception as exc:
            log.warning("%s warmup skipped: %s", name, exc)
    log.info("All models ready")


# -- Health --------------------------------------------------------------------

@app.get("/health", tags=["system"])
async def health() -> dict:
    return {"status": "ok", "version": "11.1", **health_status()}


# -- Transcribe (file upload) --------------------------------------------------

@app.post("/transcribe", tags=["transcription"])
async def transcribe(file: UploadFile = File(...)):
    """
    Zero-copy pipeline: upload -> PyAV -> numpy -> Whisper + Flan-T5 + T5-QG.
    Eliminates WAV serialization round-trip entirely.
    Questions generated at end with dedicated T5-QG model, terminal-only.
    """
    t_start = time.perf_counter()
    raw_bytes = await file.read()
    log.info("Received '%s'  (%d KB)", file.filename, len(raw_bytes) // 1024)

    # 1 . PyAV extraction -> numpy float32 (zero-copy) -------------------------
    t0 = time.perf_counter()
    try:
        audio = await asyncio.to_thread(extract_audio_to_numpy, raw_bytes)
    except Exception as exc:
        return JSONResponse(
            {"status": "error", "message": f"Audio extraction failed: {exc}"},
            status_code=422,
        )
    duration = audio_duration_from_array(audio)
    t_extract = _ms(t0)

    # 2 . Whisper + Flan-T5 (accepts numpy directly) ---------------------------
    t1 = time.perf_counter()
    try:
        whisper_result, correction = await asyncio.to_thread(
            transcribe_whisper_with_correction, audio
        )
    except Exception as exc:
        return JSONResponse(
            {"status": "error", "message": f"Transcription failed: {exc}"},
            status_code=500,
        )
    t_whisper = _ms(t1)
    t_total   = _ms(t_start)

    log.info(
        "Done -- extract:%dms  whisper+flan:%dms  total:%dms  "
        "critique(corrected:%d/kept:%d)",
        t_extract, t_whisper, t_total,
        correction.get("critique_stats", {}).get("corrected", 0),
        correction.get("critique_stats", {}).get("kept", 0),
    )

    return {
        "status": "ok",
        "filename": file.filename,
        "duration_sec": round(duration, 2),
        "whisper": whisper_result,
        "correction": correction,
        "timing": {
            "extract_ms":  t_extract,
            "whisper_ms":  t_whisper,
            "total_ms":    t_total,
        },
    }


# -- Live WebSocket ------------------------------------------------------------

@app.websocket("/ws/live")
async def ws_live(ws: WebSocket) -> None:
    """
    Live recording endpoint. Pre-allocated buffer, zero-copy path.
    Flan-T5 skipped by default on live. T5-QG questions at end, terminal-only.
    """
    await ws.accept()
    log.info("Live WebSocket connected")

    # Pre-allocate buffer: 5 minutes of 16kHz mono = ~9.6MB
    MAX_PCM    = 16_000 * 2 * 300   # sr * bytes_per_sample * seconds
    pcm_buffer = bytearray(MAX_PCM)
    buf_pos    = 0

    try:
        while True:
            try:
                data = await asyncio.wait_for(
                    ws.receive_bytes(),
                    timeout=config.WS_LIVE_TIMEOUT,
                )
            except asyncio.TimeoutError:
                log.warning("Live WebSocket timed out after %ds", config.WS_LIVE_TIMEOUT)
                await _safe_send(ws, {"type": "error", "msg": "Recording timeout"})
                await ws.close()
                return

            # -- End-of-recording sentinel -------------------------------------
            if data == b"__END__":
                await _safe_send(ws, {"type": "status", "msg": "Running Whisper..."})

                active_pcm = pcm_buffer[:buf_pos]
                if buf_pos > 3_200:   # > 0.1 s
                    audio = np.frombuffer(active_pcm, dtype=np.int16)
                    audio = audio.astype(np.float32, copy=False) * (1.0 / 32768.0)

                    try:
                        if config.FLAN_ENABLED_LIVE:
                            whisper_r, correction = await asyncio.to_thread(
                                transcribe_whisper_with_correction, audio
                            )
                        else:
                            whisper_r  = await asyncio.to_thread(transcribe_whisper, audio)
                            correction = _empty_correction(whisper_r["text"])
                            _log_questions_terminal(whisper_r["text"], source="live")

                    except Exception as exc:
                        log.error("Live final error: %s", exc)
                        whisper_r  = {"text": "", "word_count": 0, "segments": []}
                        correction = _empty_correction("")
                else:
                    whisper_r  = {"text": "", "word_count": 0, "segments": []}
                    correction = _empty_correction("")

                await _safe_send(ws, {
                    "type":       "final",
                    "whisper":    whisper_r,
                    "correction": correction,
                })
                break

            # -- Streaming chunk -----------------------------------------------
            if buf_pos + len(data) > MAX_PCM:
                await _safe_send(ws, {"type": "error", "msg": "Recording too long"})
                await ws.close()
                return
            pcm_buffer[buf_pos:buf_pos + len(data)] = data
            buf_pos += len(data)
            await _safe_send(ws, {"type": "partial", "text": "Recording... speak now"})

    except WebSocketDisconnect:
        log.info("Live WebSocket disconnected")
    except Exception as exc:
        log.error("WebSocket error: %s", exc)
        await _safe_send(ws, {"type": "error", "msg": str(exc)})


# -- Static frontend -----------------------------------------------------------

_frontend = Path(__file__).parent.parent / "frontend"
if _frontend.exists():
    app.mount("/", StaticFiles(directory=str(_frontend), html=True), name="static")


# -- Helpers -------------------------------------------------------------------

def _ms(t: float) -> int:
    return round((time.perf_counter() - t) * 1000)


async def _safe_send(ws, msg: dict):
    try:
        await ws.send_json(msg)
    except Exception:
        pass


def _empty_correction(text: str) -> dict:
    return {
        "corrected":     text,
        "enabled":       False,
        "model":         None,
        "latency_ms":    0,
        "critique_stats": {"corrected": 0, "kept": 0, "total": 0},
    }