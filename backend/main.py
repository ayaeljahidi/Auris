"""Vosper Web — FastAPI application entry point"""
import asyncio
import json
import logging
import time
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

import vosk

from .audio import extract_audio, read_wav, pcm_to_wav, audio_duration_seconds
from .models import load_vosk, load_whisper, load_marblenet, load_flan, health_status
from .transcribe import transcribe_vosk, transcribe_whisper, correct_text
from .vad import run_vad

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)-8s]  %(name)s — %(message)s",
)
log = logging.getLogger("vosper")

# ── App ────────────────────────────────────────────────────────────────────────

app = FastAPI(title="Vosper Web API", version="8.0", docs_url="/api/docs")

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
    """Pre-load all models in the background so the first request is fast."""
    loop = asyncio.get_event_loop()
    log.info("Pre-loading models…")
    for loader, name in [
        (load_vosk,      "Vosk"),
        (load_whisper,   "faster-whisper"),
        (load_marblenet, "MarbleNet VAD"),
        (load_flan,      "Flan-T5"),
    ]:
        try:
            await loop.run_in_executor(None, loader)
        except Exception as exc:
            log.warning("%s warmup skipped: %s", name, exc)
    log.info("✓ All models ready")


# ── Health ─────────────────────────────────────────────────────────────────────

@app.get("/health", tags=["system"])
async def health() -> dict:
    return {"status": "ok", "version": "8.0", **health_status()}


# ── Transcribe (file upload) ───────────────────────────────────────────────────

@app.post("/transcribe", tags=["transcription"])
async def transcribe(file: UploadFile = File(...)):
    """
    Full pipeline:  upload → FFmpeg → MarbleNet VAD → Vosk ∥ Whisper

    Total latency = t(FFmpeg) + t(VAD) + max(t(Vosk), t(Whisper))
    """
    t_start   = time.perf_counter()
    raw_bytes = await file.read()
    log.info("Received '%s'  (%d KB)", file.filename, len(raw_bytes) // 1024)

    # 1 · FFmpeg ----------------------------------------------------------------
    t0 = time.perf_counter()
    try:
        wav_bytes = extract_audio(raw_bytes)
    except Exception as exc:
        return JSONResponse(
            {"status": "error", "message": f"Audio extraction failed: {exc}"},
            status_code=422,
        )
    pcm, sr    = read_wav(wav_bytes)
    duration   = audio_duration_seconds(pcm, sr)
    t_ffmpeg   = _ms(t0)

    # 2 · VAD ------------------------------------------------------------------
    t1 = time.perf_counter()
    try:
        vad_segments, speech_wav = run_vad(wav_bytes, sr)
    except Exception as exc:
        log.error("VAD error: %s", exc)
        vad_segments, speech_wav = [], wav_bytes
    t_vad = _ms(t1)

    # 3 · Vosk + Whisper in parallel -------------------------------------------
    t2   = time.perf_counter()
    loop = asyncio.get_event_loop()
    try:
        vosk_result, whisper_result = await asyncio.gather(
            loop.run_in_executor(None, transcribe_vosk,    speech_wav),
            loop.run_in_executor(None, transcribe_whisper, speech_wav),
        )
    except Exception as exc:
        return JSONResponse(
            {"status": "error", "message": f"Transcription failed: {exc}"},
            status_code=500,
        )
    t_parallel = _ms(t2)

    # 4 · Flan-T5 correction ---------------------------------------------------
    t3 = time.perf_counter()
    loop = asyncio.get_event_loop()
    correction = await loop.run_in_executor(
        None, correct_text, whisper_result["text"]
    )
    t_correction = _ms(t3)

    t_total = _ms(t_start)

    log.info(
        "Done — ffmpeg:%dms  vad:%dms  parallel:%dms  correction:%dms  total:%dms",
        t_ffmpeg, t_vad, t_parallel, t_correction, t_total,
    )

    return {
        "status":       "ok",
        "filename":     file.filename,
        "duration_sec": round(duration, 2),
        "vad_segments": vad_segments,
        "vosk":         vosk_result,
        "whisper":      whisper_result,
        "correction":   correction,
        "timing": {
            "ffmpeg_ms":     t_ffmpeg,
            "vad_ms":        t_vad,
            "parallel_ms":   t_parallel,
            "correction_ms": t_correction,
            "total_ms":      t_total,
        },
    }


# ── Live WebSocket ─────────────────────────────────────────────────────────────

@app.websocket("/ws/live")
async def ws_live(ws: WebSocket) -> None:
    """
    Live recording endpoint.

    Protocol:
      client  →  raw 16-bit mono PCM chunks (16 kHz)
      client  →  b"__END__"          (signals end of recording)
      server  →  {"type":"partial",  "text": …}
      server  →  {"type":"final",    "whisper": …, "vad_segments": …, "vosk_text": …}
      server  →  {"type":"error",    "msg": …}
      server  →  {"type":"status",   "msg": …}
    """
    await ws.accept()
    log.info("Live WebSocket connected")

    rec = vosk.KaldiRecognizer(load_vosk(), 16_000)
    rec.SetWords(True)
    pcm_buffer = bytearray()

    try:
        while True:
            data = await ws.receive_bytes()

            # ── End-of-recording sentinel ──────────────────────────────────────
            if data == b"__END__":
                await ws.send_json({"type": "status", "msg": "Running VAD + Whisper…"})
                full_wav = pcm_to_wav(bytes(pcm_buffer), 16_000)

                if len(pcm_buffer) > 3_200:   # > 0.1 s of audio
                    loop = asyncio.get_event_loop()
                    try:
                        vad_segs, speech_wav = run_vad(full_wav, 16_000)
                        vosk_final = json.loads(rec.FinalResult())
                        vosk_text  = vosk_final.get("text", "")
                        whisper_r  = await loop.run_in_executor(
                            None, transcribe_whisper, speech_wav
                        )
                        correction = await loop.run_in_executor(
                            None, correct_text, whisper_r["text"]
                        )
                    except Exception as exc:
                        log.error("Live final error: %s", exc)
                        whisper_r  = {"text": "", "word_count": 0, "segments": []}
                        vad_segs   = []
                        vosk_text  = ""
                        correction = {"corrected": "", "enabled": False, "model": None, "latency_ms": 0}
                else:
                    whisper_r  = {"text": "", "word_count": 0, "segments": []}
                    vad_segs   = []
                    vosk_text  = ""
                    correction = {"corrected": "", "enabled": False, "model": None, "latency_ms": 0}

                await ws.send_json({
                    "type":         "final",
                    "whisper":      whisper_r,
                    "vad_segments": vad_segs,
                    "vosk_text":    vosk_text,
                    "correction":   correction,
                })
                break

            # ── Streaming partial recognition ──────────────────────────────────
            pcm_buffer.extend(data)
            if rec.AcceptWaveform(bytes(data)):
                partial = json.loads(rec.Result())
                if partial.get("text"):
                    await ws.send_json({"type": "partial", "text": partial["text"]})

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