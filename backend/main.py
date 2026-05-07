"""Auris — FastAPI application (4-way pipeline: Whisper + Flan + Emotion + QG)"""
import asyncio
import logging
import queue
import time
from concurrent.futures import ThreadPoolExecutor, Future
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
    _whisper_producer,
    _flan_consumer,
    _run_emotion,
    _run_qg,
    _SENTINEL,
)
from .emotion import detect_emotion_global, PERSISTENT_SESSION
from . import config

# Warm up Ollama on startup (optional — non-blocking)
try:
    from .qwen_questions import _load_model as _warmup_ollama
    _OLLAMA_AVAILABLE = True
except ImportError:
    _OLLAMA_AVAILABLE = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)-8s]  %(name)s — %(message)s",
)
log = logging.getLogger("auris")

app = FastAPI(title="Auris API", version="17.0-qg", docs_url="/api/docs")

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
    """Pre-load ALL models in background threads."""
    log.info("=" * 60)
    log.info("🚀 Starting Auris v17 — 4-way pipeline (Whisper+Flan+Emotion+QG)")
    log.info("=" * 60)

    log.info("Pre-loading Whisper model...")
    await asyncio.to_thread(load_whisper)

    log.info("Pre-loading Flan-T5 model...")
    await asyncio.to_thread(load_flan)

    log.info("Pre-loading Emotion ONNX model...")
    await asyncio.to_thread(load_emotion_model)

    if PERSISTENT_SESSION is not None:
        log.info("✓ Emotion ONNX session loaded and ready (persistent)")
    else:
        log.warning("⚠ Emotion session not available (disabled or error)")

    if _OLLAMA_AVAILABLE:
        log.info("Warming up Ollama / Qwen QG model (background)…")
        asyncio.create_task(asyncio.to_thread(_warmup_ollama))
    else:
        log.warning("⚠ Qwen QG not available — install Ollama + qwen2.5:1.5b")

    log.info("✓ All models ready — server starting")
    log.info("=" * 60)


# ── Health ─────────────────────────────────────────────────────────────────────

@app.get("/health", tags=["system"])
async def health() -> dict:
    return {
        "status": "ok",
        "version": "17.0-qg",
        "qg_available": _OLLAMA_AVAILABLE,
        **health_status(),
    }


# ── Emotion-only endpoint (unchanged) ─────────────────────────────────────────

@app.post("/emotion", tags=["emotion"])
async def emotion_global(file: UploadFile = File(...)):
    """Detect emotion on full audio — no Whisper, no QG."""
    t_start = time.perf_counter()
    raw_bytes = await file.read()
    log.info("Received: %s (%d KB)", file.filename, len(raw_bytes) // 1024)

    try:
        audio = await asyncio.to_thread(extract_audio_to_numpy, raw_bytes)
    except Exception as exc:
        return JSONResponse({"error": f"Audio extraction failed: {exc}"}, status_code=422)

    duration  = audio_duration_from_array(audio)
    t_extract = _ms(t_start)

    try:
        result = await asyncio.to_thread(detect_emotion_global, audio)
    except Exception as exc:
        return JSONResponse({"error": f"Emotion detection failed: {exc}"}, status_code=500)

    result["filename"]    = file.filename
    result["duration_sec"] = round(duration, 2)
    result["extract_ms"]  = t_extract
    return result


# ── /transcribe — 4-way parallel pipeline ────────────────────────────────────

@app.post("/transcribe", tags=["transcription"])
async def transcribe(file: UploadFile = File(...)):
    """
    Full pipeline: Whisper + Flan-T5 + Emotion ONNX + Qwen QG.

    Returns:
        whisper      — raw transcript + segments
        correction   — Flan-corrected text + stats
        emotion      — dominant emotion + confidence
        questions    — 4 jury-style questions from Qwen
        timing       — per-stage latency breakdown
    """
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
    duration  = audio_duration_from_array(audio)
    t_extract = _ms(t0)

    t1 = time.perf_counter()
    try:
        whisper_result, correction, emotion, questions = await asyncio.to_thread(
            transcribe_whisper_with_correction_and_emotion, audio, True
        )
    except Exception as exc:
        log.error("Transcription failed: %s", exc)
        return JSONResponse(
            {"status": "error", "message": f"Transcription failed: {exc}"},
            status_code=500,
        )
    t_pipeline = _ms(t1)
    t_total    = _ms(t_start)

    _print_emotion_terminal(emotion)
    _print_qg_terminal(questions)

    log.info(
        "Done — extract:%dms  pipeline:%dms  total:%dms  emotion:%s  qg:%d questions",
        t_extract, t_pipeline, t_total,
        emotion.get("emotion", "n/a"),
        len(questions.get("questions", [])),
    )

    return {
        "status":       "ok",
        "filename":     file.filename,
        "duration_sec": round(duration, 2),
        "whisper":      whisper_result,
        "correction":   correction,
        "emotion":      emotion,
        "questions":    questions,
        "timing": {
            "extract_ms":  t_extract,
            "pipeline_ms": t_pipeline,
            "total_ms":    t_total,
        },
    }


# ── Live WebSocket — 4-way pipeline (QG skipped in real-time mode) ────────────
#
#  QG requires the FULL corrected transcript and takes several seconds on CPU.
#  In live mode we skip it by default (run_qg=False) to minimise end-to-end
#  latency.  Set QG_ENABLED_LIVE=true env var to enable it anyway.
#
#  Pipeline timeline (live, QG off):
#    [=====Whisper======]
#            [=Flan s1=][=s2=]…
#    [==========Emotion===========]
#                                  ^ final sent to client
#
#  Pipeline timeline (live, QG on):
#    [=====Whisper======]
#            [=Flan s1=][=s2=]…
#    [==========Emotion===========]
#                                [==Qwen QG==]  ← extra wait
#                                               ^ final sent to client

import os as _os
_QG_ENABLED_LIVE = _os.environ.get("QG_ENABLED_LIVE", "false").lower() == "true"


def _live_pipeline(audio: np.ndarray) -> tuple[dict, dict, dict, dict]:
    """
    4-way pipeline for the WebSocket live endpoint.
    QG is controlled by QG_ENABLED_LIVE env var (default: off).
    """
    t0 = time.perf_counter()

    MAX_SEGS   = 256
    results    = [None] * MAX_SEGS
    flan_stats = {"corrected": 0, "kept": 0}
    seg_queue: queue.Queue = queue.Queue(maxsize=4)

    # Respect FLAN_ENABLED_LIVE flag
    original_flan = config.FLAN_ENABLED
    config.FLAN_ENABLED = config.FLAN_ENABLED_LIVE

    try:
        with ThreadPoolExecutor(max_workers=4) as executor:
            future_whisper = executor.submit(_whisper_producer, audio, seg_queue)
            future_flan    = executor.submit(_flan_consumer, seg_queue, results, flan_stats)
            future_emotion = executor.submit(_run_emotion, audio)

            future_whisper.result()
            future_flan.result()

            # Build corrected text before launching QG
            ordered_partial = [r for r in results if r is not None]
            corrected_text  = " ".join(c for _, _, c in ordered_partial).strip()

            future_qg: Future = (
                executor.submit(_run_qg, corrected_text)
                if _QG_ENABLED_LIVE
                else executor.submit(lambda: {
                    "enabled": False, "questions": [], "raw": "",
                    "latency_ms": 0, "error": "skipped (live mode)",
                })
            )

            emotion_data = future_emotion.result()
            qg_data      = future_qg.result()

    finally:
        config.FLAN_ENABLED = original_flan

    t_pipeline = round((time.perf_counter() - t0) * 1000)

    ordered = [r for r in results if r is not None]
    segments, raw_parts, corrected_parts = [], [], []
    for seg_dict, raw, corrected in ordered:
        segments.append(seg_dict)
        raw_parts.append(raw)
        corrected_parts.append(corrected)

    text           = " ".join(raw_parts).strip()
    corrected_text = " ".join(corrected_parts).strip()
    total_segs     = len(segments)

    whisper_result = {
        "text":       text,
        "word_count": len(text.split()) if text else 0,
        "segments":   segments,
    }
    correction_result = {
        "corrected":  corrected_text,
        "enabled":    config.FLAN_ENABLED_LIVE,
        "model":      config.FLAN_MODEL if config.FLAN_ENABLED_LIVE else None,
        "latency_ms": t_pipeline,
        "critique_stats": {
            "corrected": flan_stats["corrected"],
            "kept":      flan_stats["kept"],
            "total":     total_segs,
        },
    }
    emotion_result = {
        "enabled":         emotion_data.get("enabled", config.EMOTION_ENABLED),
        "emotion":         emotion_data.get("emotion", "unknown"),
        "confidence":      emotion_data.get("confidence", 0.0),
        "latency_ms":      emotion_data.get("latency_ms", 0),
        "is_reliable":     emotion_data.get("is_reliable", False),
        "all_probs":       emotion_data.get("all_probs", {}),
        "realtime_factor": emotion_data.get("realtime_factor", 0),
        "inference_ms":    emotion_data.get("inference_ms", 0),
    }
    questions_result = {
        "enabled":    qg_data.get("enabled", False),
        "questions":  qg_data.get("questions", []),
        "raw":        qg_data.get("raw", ""),
        "latency_ms": qg_data.get("latency_ms", 0),
        "error":      qg_data.get("error"),
    }

    log.info(
        "✅ Live pipeline — whisper:%d seg | flan(live):%s +%d/=%d | "
        "emotion:%s(%.1f%%) | qg:%s | total:%dms",
        total_segs,
        "ON" if config.FLAN_ENABLED_LIVE else "OFF",
        flan_stats["corrected"], flan_stats["kept"],
        emotion_result["emotion"], emotion_result["confidence"] * 100,
        f"{len(questions_result['questions'])} questions" if questions_result["enabled"] else "skipped",
        t_pipeline,
    )

    return whisper_result, correction_result, emotion_result, questions_result


@app.websocket("/ws/live")
async def ws_live(ws: WebSocket) -> None:
    """
    Live recording WebSocket endpoint.

    Client protocol (unchanged):
        → raw PCM bytes (int16, mono, 16 kHz) while recording
        → b"__END__" when mic stops

    Server responses:
        ← {"type": "partial",  "text": "Recording… speak now"}
        ← {"type": "status",   "msg": "Running pipeline…"}
        ← {"type": "final",    "whisper": …, "correction": …,
                                "emotion": …, "questions": …,
                                "duration_sec": …, "pipeline_ms": …}
        ← {"type": "error",    "msg": "…"}
    """
    await ws.accept()
    log.info("Live WebSocket connected")

    MAX_PCM    = 16_000 * 2 * 300
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

            if data == b"__END__":
                await _safe_send(ws, {"type": "status", "msg": "Running pipeline…"})

                active_pcm = pcm_buffer[:buf_pos]

                if buf_pos > 3_200:
                    audio = np.frombuffer(active_pcm, dtype=np.int16)
                    audio = audio.astype(np.float32, copy=False) * (1.0 / 32768.0)
                    duration_sec = audio.size / 16_000

                    try:
                        whisper_r, correction, emotion, questions = await asyncio.to_thread(
                            _live_pipeline, audio
                        )
                    except Exception as exc:
                        log.error("Live pipeline error: %s", exc)
                        await _safe_send(ws, {"type": "error", "msg": str(exc)})
                        break

                    _print_emotion_terminal(emotion)
                    _print_qg_terminal(questions)

                else:
                    duration_sec = buf_pos / (16_000 * 2)
                    whisper_r  = {"text": "", "word_count": 0, "segments": []}
                    correction = _empty_correction("")
                    emotion    = _empty_emotion()
                    questions  = _empty_questions()

                await _safe_send(ws, {
                    "type":         "final",
                    "whisper":      whisper_r,
                    "correction":   correction,
                    "emotion":      emotion,
                    "questions":    questions,
                    "duration_sec": round(duration_sec, 2),
                    "pipeline_ms":  correction.get("latency_ms", 0),
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
        "corrected":  text,
        "enabled":    False,
        "model":      None,
        "latency_ms": 0,
        "critique_stats": {"corrected": 0, "kept": 0, "total": 0},
    }


def _empty_emotion() -> dict:
    return {
        "enabled":         False,
        "emotion":         "unknown",
        "confidence":      0.0,
        "latency_ms":      0,
        "is_reliable":     False,
        "all_probs":       {},
        "realtime_factor": 0,
        "inference_ms":    0,
    }


def _empty_questions() -> dict:
    return {
        "enabled":    False,
        "questions":  [],
        "raw":        "",
        "latency_ms": 0,
        "error":      "audio too short",
    }


def _print_emotion_terminal(emotion: dict) -> None:
    """Print emotion results to terminal with a styled box."""
    e     = emotion.get("emotion", "unknown").upper()
    conf  = emotion.get("confidence", 0.0)
    lat   = emotion.get("latency_ms", 0)
    rel   = "✓ RELIABLE" if emotion.get("is_reliable") else "⚠ LOW CONFIDENCE"

    print()
    print("╔" + "═" * 60 + "╗")
    print("║" + " 🎭  EMOTION DETECTION".center(60) + "║")
    print("╠" + "═" * 60 + "╣")

    if emotion.get("enabled") is False:
        print("║" + "  Emotion detection disabled or unavailable".ljust(60) + "║")
        print("╚" + "═" * 60 + "╝")
        return

    print(f"║  Emotion   : {e:<15}  {rel}".ljust(61) + "║")
    print(f"║  Confidence: {conf:.1%}".ljust(61) + "║")
    print(f"║  Latency   : {lat}ms".ljust(61) + "║")

    all_probs = emotion.get("all_probs", {})
    if all_probs:
        print("╠" + "─" * 60 + "╣")
        for label, prob in sorted(all_probs.items(), key=lambda x: -x[1]):
            bar = "█" * int(prob * 20)
            print(f"║  {label:<10} {prob:>6.1%}  {bar}".ljust(61) + "║")

    print("╚" + "═" * 60 + "╝")
    log.info("🎭 Emotion: %s (%.1f%%) | %dms", e, conf * 100, lat)


def _print_qg_terminal(questions: dict) -> None:
    """Print generated questions to terminal with a styled box."""
    print()
    print("╔" + "═" * 60 + "╗")
    print("║" + " 🧠  JURY QUESTIONS (Qwen QG)".center(60) + "║")
    print("╠" + "═" * 60 + "╣")

    if not questions.get("enabled") or questions.get("error"):
        reason = questions.get("error") or "QG disabled"
        print(f"║  {reason[:58]}".ljust(61) + "║")
        print("╚" + "═" * 60 + "╝")
        return

    lat  = questions.get("latency_ms", 0)
    qs   = questions.get("questions", [])

    print(f"║  Generated {len(qs)} questions  |  Latency: {lat}ms".ljust(61) + "║")
    print("╠" + "─" * 60 + "╣")
    for i, q in enumerate(qs, 1):
        # Word-wrap at 56 chars
        words  = q.split()
        lines  = []
        current = ""
        for w in words:
            if len(current) + len(w) + 1 > 56:
                lines.append(current)
                current = w
            else:
                current = (current + " " + w).strip()
        if current:
            lines.append(current)

        first = True
        for ln in lines:
            prefix = f"  {i}. " if first else "     "
            print(f"║{prefix}{ln}".ljust(61) + "║")
            first = False
        if len(qs) > 1 and i < len(qs):
            print("║" + " " * 60 + "║")

    print("╚" + "═" * 60 + "╝")
    log.info("🧠 QG: %d questions generated in %dms", len(qs), lat)