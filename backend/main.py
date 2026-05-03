"""Auris — FastAPI application (émotion globale sur audio complet)"""
import asyncio
import logging
import time
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from .audio import extract_audio_to_numpy, audio_duration_from_array
from .models import load_whisper, load_flan, load_emotion_model, health_status
from .transcribe import (
    transcribe_whisper,
    transcribe_whisper_with_correction,
    transcribe_whisper_with_correction_and_emotion,
)
from .emotion import detect_emotion_global, PERSISTENT_SESSION
from . import config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)-8s]  %(name)s — %(message)s",
)
log = logging.getLogger("auris")

app = FastAPI(title="Auris API", version="16.0-onnx-community", docs_url="/api/docs")

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
    """Pre-load ALL models in background (ONNX loaded once and cached)."""
    log.info("=" * 60)
    log.info("🚀 Starting Auris with ONNX Community pre-converted model")
    log.info("=" * 60)

    log.info("Pre-loading Whisper model...")
    await asyncio.to_thread(load_whisper)

    log.info("Pre-loading Flan-T5 model...")
    await asyncio.to_thread(load_flan)

    log.info("Pre-loading Emotion ONNX model from HF community...")
    await asyncio.to_thread(load_emotion_model)

    if PERSISTENT_SESSION is not None:
        log.info("✓ Emotion ONNX session loaded and ready (persistent)")
    else:
        log.warning("⚠ Emotion session not available (disabled or error)")

    log.info("✓ All models ready - server starting")
    log.info("=" * 60)


# ── Health ─────────────────────────────────────────────────────────────────────

@app.get("/health", tags=["system"])
async def health() -> dict:
    return {"status": "ok", "version": "16.0-onnx-community", **health_status()}


# ═══════════════════════════════════════════════════════════════════════════════
# 🆕 NOUVEAU ENDPOINT: Émotion globale (sans Whisper, sans découpage)
# ═══════════════════════════════════════════════════════════════════════════════

@app.post("/emotion", tags=["emotion"])
async def emotion_global(file: UploadFile = File(...)):
    """
    Détecte l'émotion sur l'AUDIO COMPLET.
    - Pas de découpage
    - Pas de Whisper
    - Pas de timeline
    - Une seule émotion globale
    - Session ONNX persistante (pas de rechargement)

    Exemple:
        curl -X POST "http://localhost:8000/emotion" -F "file=@audio.mp3"
    """
    t_start = time.perf_counter()
    raw_bytes = await file.read()
    log.info(f"Received: {file.filename} ({len(raw_bytes)//1024} KB)")

    try:
        audio = await asyncio.to_thread(extract_audio_to_numpy, raw_bytes)
    except Exception as exc:
        return JSONResponse(
            {"error": f"Audio extraction failed: {exc}"},
            status_code=422
        )

    duration = audio_duration_from_array(audio)
    t_extract = round((time.perf_counter() - t_start) * 1000)

    try:
        result = await asyncio.to_thread(detect_emotion_global, audio)
    except Exception as exc:
        return JSONResponse(
            {"error": f"Emotion detection failed: {exc}"},
            status_code=500
        )

    result["filename"] = file.filename
    result["duration_sec"] = round(duration, 2)
    result["extract_ms"] = t_extract

    log.debug(f"Emotion done: {duration:.1f}s audio → {result['latency_ms']}ms ({result['realtime_factor']:.1f}x realtime)")

    return result


# ── Transcribe (avec Whisper + émotion par segment) ───────────────────────────

@app.post("/transcribe", tags=["transcription"])
async def transcribe(file: UploadFile = File(...)):
    """Transcription complète (Whisper + correction + émotion par segment)"""
    t_start = time.perf_counter()
    raw_bytes = await file.read()
    log.info("Received '%s'  (%d KB)", file.filename, len(raw_bytes) // 1024)

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

    t1 = time.perf_counter()
    try:
        whisper_result, correction, emotion = await asyncio.to_thread(
            transcribe_whisper_with_correction_and_emotion, audio
        )
    except Exception as exc:
        log.error(f"Transcription failed: {exc}")
        return JSONResponse(
            {"status": "error", "message": f"Transcription failed: {exc}"},
            status_code=500,
        )
    t_whisper = _ms(t1)
    t_total = _ms(t_start)

    _print_emotion_terminal(emotion)

    log.info(
        "Done — extract:%dms  pipeline:%dms  total:%dms  emotion:%s",
        t_extract, t_whisper, t_total,
        emotion.get("emotion", "n/a"),
    )

    return {
        "status": "ok",
        "filename": file.filename,
        "duration_sec": round(duration, 2),
        "whisper": whisper_result,
        "correction": correction,
        "emotion": emotion,
        "timing": {
            "extract_ms": t_extract,
            "whisper_ms": t_whisper,
            "total_ms": t_total,
        },
    }


# ── Live WebSocket ─────────────────────────────────────────────────────────────

@app.websocket("/ws/live")
async def ws_live(ws: WebSocket) -> None:
    """Live recording endpoint."""
    await ws.accept()
    log.info("Live WebSocket connected")

    MAX_PCM = 16_000 * 2 * 300
    pcm_buffer = bytearray(MAX_PCM)
    buf_pos = 0

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

            if data == b"__END__":
                await _safe_send(ws, {"type": "status", "msg": "Running Whisper…"})

                active_pcm = pcm_buffer[:buf_pos]
                if buf_pos > 3_200:
                    audio = np.frombuffer(active_pcm, dtype=np.int16)
                    audio = audio.astype(np.float32, copy=False) * (1.0 / 32768.0)

                    try:
                        if config.FLAN_ENABLED_LIVE:
                            whisper_r, correction = await asyncio.to_thread(
                                transcribe_whisper_with_correction, audio
                            )
                        else:
                            whisper_r = await asyncio.to_thread(
                                transcribe_whisper, audio
                            )
                            correction = _empty_correction(whisper_r["text"])
                    except Exception as exc:
                        log.error("Live final error: %s", exc)
                        whisper_r = {"text": "", "word_count": 0, "segments": []}
                        correction = _empty_correction("")
                else:
                    whisper_r = {"text": "", "word_count": 0, "segments": []}
                    correction = _empty_correction("")

                await _safe_send(ws, {
                    "type": "final",
                    "whisper": whisper_r,
                    "correction": correction,
                })
                break

            if buf_pos + len(data) > MAX_PCM:
                await _safe_send(ws, {"type": "error", "msg": "Recording too long"})
                await ws.close()
                return
            pcm_buffer[buf_pos:buf_pos + len(data)] = data
            buf_pos += len(data)
            await _safe_send(ws, {"type": "partial", "text": "Recording… speak now"})

    except WebSocketDisconnect:
        log.info("Live WebSocket disconnected")
    except Exception as exc:
        log.error("WebSocket error: %s", exc)
        await _safe_send(ws, {"type": "error", "msg": str(exc)})


# ── Static frontend ────────────────────────────────────────────────────────────

_frontend = Path(__file__).parent.parent / "frontend"
if _frontend.exists():
    app.mount("/", StaticFiles(directory=str(_frontend), html=True), name="static")


# ── Helpers ────────────────────────────────────────────────────────────────────

def _ms(t: float) -> int:
    return round((time.perf_counter() - t) * 1000)


async def _safe_send(ws, msg: dict):
    try:
        await ws.send_json(msg)
    except Exception:
        pass


def _empty_correction(text: str) -> dict:
    return {
        "corrected": text,
        "enabled": False,
        "model": None,
        "latency_ms": 0,
        "critique_stats": {"corrected": 0, "kept": 0, "total": 0},
    }


def _print_emotion_terminal(emotion: dict) -> None:
    """Affiche les résultats d'émotion globale dans le terminal."""
    print()
    print("╔" + "═" * 60 + "╗")
    print("║" + " 🎭  DETECTION D'EMOTION (ONNX Community)".center(60) + "║")
    print("╠" + "═" * 60 + "╣")

    if emotion.get("enabled") is False:
        print("║" + "  Détection d'émotions désactivée".ljust(60) + "║")
        print("╚" + "═" * 60 + "╝")
        return

    emotion_label = emotion.get("emotion", "unknown").upper()
    confidence = emotion.get("confidence", 0.0)
    latency = emotion.get("latency_ms", 0)

    print(f"║  Émotion détectée:  {emotion_label:15s}                              ║")
    print(f"║  Confiance:         {confidence:.1%}                                      ║")
    print(f"║  Temps d'analyse:   {latency}ms                                        ║")
    print("╚" + "═" * 60 + "╝")

    log.info(f"🎭 Emotion: {emotion_label} ({confidence:.1%}) in {latency}ms")