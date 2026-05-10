"""Auris — FastAPI application (OPTIMIZED parallel pipeline)

Pipeline: Whisper || Audio Emotion (T=0)
          → Text Emotion || Flan (when Whisper done)
          → Fusion || QG (when Phase 2 done)
"""
import asyncio
import logging
import queue
import time
from concurrent.futures import ThreadPoolExecutor, Future, wait as _wait, ALL_COMPLETED
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from .audio import extract_audio_to_numpy, audio_duration_from_array
from .models import load_whisper, load_flan, load_emotion_model, load_text_emotion_model, health_status
from .transcribe import (
    transcribe_whisper,
    transcribe_whisper_with_correction,
    transcribe_whisper_with_correction_and_emotion,
    _whisper_producer,
    _flan_consumer,
    _run_audio_emotion,
    _run_text_emotion,
    _run_qg,
    _SENTINEL,
    FUSION_AVAILABLE,
    _correct_batch,
    _chunk_by_tokens,
)
from .emotion import detect_emotion_global
from .emotion_fusion import fuse_emotions
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

app = FastAPI(title="Auris API", version="18.1-optimized", docs_url="/api/docs")

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
    log.info("Starting Auris v18.1 — Optimized Parallel Pipeline")

    log.info("Pre-loading Whisper model...")
    await asyncio.to_thread(load_whisper)

    log.info("Pre-loading Flan-T5 model...")
    await asyncio.to_thread(load_flan)

    log.info("Pre-loading Wav2Vec2 Emotion model (8 classes)...")
    await asyncio.to_thread(load_emotion_model)

    log.info("Pre-loading DistilRoBERTa Text Emotion model (7 classes)...")
    await asyncio.to_thread(load_text_emotion_model)

    if _OLLAMA_AVAILABLE:
        log.info("Warming up Ollama / Qwen QG model (background)...")
        asyncio.create_task(asyncio.to_thread(_warmup_ollama))
    else:
        log.warning("⚠ Qwen QG not available — install Ollama + qwen2.5:1.5b")

    log.info("✓ All models ready — server starting")
    log.info("Emotion classes: angry, disgust, fear, happy, neutral, sad, surprised")
    log.info("=" * 60)


# ── Health ─────────────────────────────────────────────────────────────────────

@app.get("/health", tags=["system"])
async def health() -> dict:
    return {
        "status": "ok",
        "version": "18.1-optimized",
        "qg_available": _OLLAMA_AVAILABLE,
        "emotion_pipeline": "fusion (wav2vec2 + distilroberta)",
        "audio_emotion_classes": ["angry", "calm", "disgust", "fear", "happy", "neutral", "sad", "surprised"],
        "text_emotion_classes":  ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"],
        **health_status(),
    }


# ── Emotion-only endpoint ─────────────────────────────────────────────────────

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

    duration = audio_duration_from_array(audio)
    t_extract = _ms(t_start)

    try:
        result = await asyncio.to_thread(detect_emotion_global, audio, config.EMOTION_SR)
    except Exception as exc:
        return JSONResponse({"error": f"Emotion detection failed: {exc}"}, status_code=500)

    result["filename"] = file.filename
    result["duration_sec"] = round(duration, 2)
    result["extract_ms"] = t_extract
    return result


# ── /transcribe — OPTIMIZED 5-way parallel pipeline ──────────────────────────

@app.post("/transcribe", tags=["transcription"])
async def transcribe(file: UploadFile = File(...)):
    """
    Full OPTIMIZED pipeline: Whisper + Flan-T5 + Audio Emotion + Text Emotion + QG.

    Optimized dependency graph:
      Phase 1 (T=0):    Whisper ASR  ||  Audio Emotion (Wav2Vec2)
      Phase 2 (Whisper done): Text Emotion (raw) || Flan Correction
      Phase 3 (Phase 2 done): Emotion Fusion || Qwen QG

    Text Emotion runs on RAW text, not corrected — saves Flan latency.
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
    duration = audio_duration_from_array(audio)
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
    t_total = _ms(t_start)

    _print_emotion_terminal(emotion)
    _print_qg_terminal(questions)

    log.info(
        "Done — extract:%dms  pipeline:%dms  total:%dms  emotion:%s  qg:%d questions",
        t_extract, t_pipeline, t_total,
        emotion.get("emotion", "n/a"),
        len(questions.get("questions", [])),
    )

    return {
        "status": "ok",
        "filename": file.filename,
        "duration_sec": round(duration, 2),
        "whisper": whisper_result,
        "correction": correction,
        "emotion": emotion,
        "questions": questions,
        "timing": {
            "extract_ms":  t_extract,
            "pipeline_ms": t_pipeline,
            "total_ms":    t_total,
            "stage_ms":    correction.get("stage_ms", {}),
        },
    }


# ── Live WebSocket ────────────────────────────────────────────────────────────

import os as _os
_QG_ENABLED_LIVE = _os.environ.get("QG_ENABLED_LIVE", "true").lower() == "true"


def _live_pipeline(audio: np.ndarray) -> tuple[dict, dict, dict, dict]:
    """
    OPTIMIZED 5-way pipeline for the WebSocket live endpoint.

    Same optimized dependency graph as /transcribe:
      Phase 1: Whisper || Audio Emotion
      Phase 2: Text Emotion (raw) || Flan
      Phase 3: Fusion || QG
    """
    t0 = time.perf_counter()

    MAX_SEGS = 256
    results = [None] * MAX_SEGS
    flan_stats = {"corrected": 0, "kept": 0}
    seg_queue: queue.Queue = queue.Queue(maxsize=0)

    with ThreadPoolExecutor(max_workers=5) as executor:
        # ── PHASE 1: Whisper + Audio Emotion (parallel) ──────────────────
        future_audio_emotion = executor.submit(_run_audio_emotion, audio)
        future_whisper = executor.submit(_whisper_producer, audio, seg_queue)

        # Wait for Whisper
        future_whisper.result()
        t_whisper_ms = round((time.perf_counter() - t0) * 1000)

        # Drain queue to get raw_text
        queue_items: list = []
        raw_parts: list[str] = []
        segments: list[dict] = []
        while True:
            item = seg_queue.get()
            if item is _SENTINEL:
                break
            queue_items.append(item)
            idx, seg_dict, raw_text, needs_corr = item
            results[idx] = (seg_dict, raw_text, raw_text)
            raw_parts.append(raw_text)
            segments.append(seg_dict)

        raw_text = " ".join(raw_parts).strip()
        total_segs = len(segments)

        # ── PHASE 2: Text Emotion (raw) + Flan (parallel) ──────────────────
        t_phase2 = time.perf_counter()

        future_text_emotion = executor.submit(_run_text_emotion, raw_text)

        # Flan correction (in-thread, queue already drained)
        if config.FLAN_ENABLED:
            model, tokenizer = load_flan()
            if model is not None and tokenizer is not None:
                device = next(model.parameters()).device

                needs: list = []
                keeps: list = []
                for idx, seg_dict, raw_text, needs_correction in queue_items:
                    if needs_correction:
                        needs.append((idx, seg_dict, raw_text))
                    else:
                        keeps.append((idx, seg_dict, raw_text))
                        flan_stats["kept"] += 1

                if needs:
                    raw_texts = [raw for _, _, raw in needs]
                    corrected_s = _correct_batch(raw_texts, model, tokenizer, device)
                    for (idx, seg_dict, raw_text), corrected in zip(needs, corrected_s):
                        changed = corrected != raw_text.strip()
                        flan_stats["corrected"] += int(changed)
                        flan_stats["kept"] += int(not changed)
                        results[idx] = (seg_dict, raw_text, corrected)

                for idx, seg_dict, raw_text in keeps:
                    results[idx] = (seg_dict, raw_text, raw_text.strip())

        t_flan_ms = round((time.perf_counter() - t_phase2) * 1000)

        # Build corrected_text
        ordered = [r for r in results if r is not None]
        corrected_parts = [corrected for _, _, corrected in ordered]
        corrected_text = " ".join(corrected_parts).strip()

        # ── PHASE 3: Fusion + QG (parallel) ──────────────────────────────
        t_phase3 = time.perf_counter()

        _wait([future_audio_emotion, future_text_emotion], return_when=ALL_COMPLETED)
        audio_emotion_data = future_audio_emotion.result()
        text_emotion_data = future_text_emotion.result()

        # Fusion (fast, in-thread)
        if FUSION_AVAILABLE:
            fused_emotion = fuse_emotions(audio_emotion_data, text_emotion_data)
        else:
            fused_emotion = {
                **audio_emotion_data,
                "source": "audio_only",
                "fusion_weights": {"audio": 1.0, "text": 0.0},
                "audio_emotion": audio_emotion_data,
                "text_emotion": text_emotion_data,
            }

        # QG (in thread, async HTTP inside)
        future_qg: Future = (
            executor.submit(_run_qg, corrected_text)
            if _QG_ENABLED_LIVE
            else executor.submit(lambda: {
                "enabled": False, "questions": [], "raw": "",
                "latency_ms": 0, "error": "skipped (live mode)",
            })
        )
        qg_data = future_qg.result()

    t_pipeline = round((time.perf_counter() - t0) * 1000)

    # Build results
    whisper_result = {
        "text": raw_text,
        "word_count": len(raw_text.split()) if raw_text else 0,
        "segments": segments,
    }
    correction_result = {
        "corrected": corrected_text,
        "enabled": config.FLAN_ENABLED_LIVE,
        "model": config.FLAN_MODEL if config.FLAN_ENABLED_LIVE else None,
        "latency_ms": t_pipeline,
        "critique_stats": {
            "corrected": flan_stats["corrected"],
            "kept": flan_stats["kept"],
            "total": total_segs,
        },
    }
    questions_result = {
        "enabled": qg_data.get("enabled", False),
        "questions": qg_data.get("questions", []),
        "raw": qg_data.get("raw", ""),
        "latency_ms": qg_data.get("latency_ms", 0),
        "error": qg_data.get("error"),
    }

    log.info(
        "Live pipeline (OPTIMIZED) — whisper:%d seg | flan(live):%s +%d/=%d | "
        "audio_emotion:%s(%.1f%%) | text_emotion:%s(%.1f%%) | "
        "fused:%s(%.1f%%) | qg:%s | total:%dms",
        total_segs,
        "ON" if config.FLAN_ENABLED_LIVE else "OFF",
        flan_stats["corrected"], flan_stats["kept"],
        audio_emotion_data.get("emotion", "n/a"),
        audio_emotion_data.get("confidence", 0.0) * 100,
        text_emotion_data.get("emotion", "n/a"),
        text_emotion_data.get("confidence", 0.0) * 100,
        fused_emotion.get("emotion", "n/a"),
        fused_emotion.get("confidence", 0.0) * 100,
        f"{len(questions_result['questions'])} questions" if questions_result["enabled"] else "skipped",
        t_pipeline,
    )

    return whisper_result, correction_result, fused_emotion, questions_result


# ── WebSocket endpoint ─────────────────────────────────────────────────────────

@app.websocket("/ws/live")
async def ws_live(ws: WebSocket) -> None:
    """
    Live recording WebSocket endpoint.
    Client sends audio/webm chunks from MediaRecorder, then b"__END__".
    Server decodes the webm container and runs the OPTIMIZED pipeline.
    """
    await ws.accept()
    log.info("Live WebSocket connected")

    webm_chunks: list[bytes] = []
    total_bytes = 0
    MAX_BYTES = 300 * 1024 * 1024  # 300 MB

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
                await _safe_send(ws, {"type": "status", "msg": "Running pipeline..."})

                if total_bytes < 1024:
                    await _safe_send(ws, {
                        "type": "final",
                        "whisper": {"text": "", "word_count": 0, "segments": []},
                        "correction": _empty_correction(""),
                        "emotion": _empty_emotion(),
                        "questions": _empty_questions(),
                        "duration_sec": 0,
                        "pipeline_ms": 0,
                    })
                    break

                raw_webm = b"".join(webm_chunks)
                webm_chunks.clear()

                try:
                    audio = await asyncio.to_thread(extract_audio_to_numpy, raw_webm)
                except Exception as exc:
                    log.error("WebSocket audio decode failed: %s", exc)
                    await _safe_send(ws, {"type": "error", "msg": f"Audio decode failed: {exc}"})
                    break

                duration_sec = audio_duration_from_array(audio)

                if audio.size < 3200:
                    await _safe_send(ws, {
                        "type": "final",
                        "whisper": {"text": "", "word_count": 0, "segments": []},
                        "correction": _empty_correction(""),
                        "emotion": _empty_emotion(),
                        "questions": _empty_questions(),
                        "duration_sec": round(duration_sec, 2),
                        "pipeline_ms": 0,
                    })
                    break

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

                await _safe_send(ws, {
                    "type": "final",
                    "whisper": whisper_r,
                    "correction": correction,
                    "emotion": emotion,
                    "questions": questions,
                    "duration_sec": round(duration_sec, 2),
                    "pipeline_ms": correction.get("latency_ms", 0),
                })
                break

            if total_bytes + len(data) > MAX_BYTES:
                await _safe_send(ws, {"type": "error", "msg": "Recording too long (max 5 min)"})
                await ws.close()
                return

            webm_chunks.append(data)
            total_bytes += len(data)
            await _safe_send(ws, {"type": "partial", "text": "Recording... speak now"})

    except WebSocketDisconnect:
        log.info("Live WebSocket disconnected")
    except Exception as exc:
        log.error("WebSocket error: %s", exc)
        await _safe_send(ws, {"type": "error", "msg": str(exc)})


# ── Static frontend ────────────────────────────────────────────────────────────

_frontend = Path(__file__).parent.parent / "frontend" / "dist"
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


def _empty_emotion() -> dict:
    return {
        "enabled": False,
        "emotion": "unknown",
        "confidence": 0.0,
        "latency_ms": 0,
        "is_reliable": False,
        "all_probs": {},
        "realtime_factor": 0,
        "inference_ms": 0,
    }


def _empty_questions() -> dict:
    return {
        "enabled": False,
        "questions": [],
        "raw": "",
        "latency_ms": 0,
        "error": "audio too short",
    }


def _print_emotion_terminal(emotion: dict) -> None:
    if emotion.get("enabled") is False:
        log.info("Emotion: disabled/unavailable")
        return
    e    = emotion.get("emotion", "unknown").upper()
    conf = emotion.get("confidence", 0.0)
    lat  = emotion.get("latency_ms", 0)
    rel  = "reliable" if emotion.get("is_reliable") else "low-confidence"
    src  = emotion.get("source", "unknown")

    audio_e = emotion.get("audio_emotion", {})
    text_e  = emotion.get("text_emotion", {})
    weights = emotion.get("fusion_weights", {})

    log.info(
        "Emotion[FUSED]: %s (%.1f%%) [%s | src:%s] | %dms",
        e, conf * 100, rel, src, lat,
    )
    if audio_e:
        log.info(
            "  ↳ Audio  (w=%.0f%%): %s (%.1f%%)",
            weights.get("audio", 0) * 100,
            audio_e.get("emotion", "n/a").upper(),
            audio_e.get("confidence", 0.0) * 100,
        )
    if text_e:
        log.info(
            "  ↳ Text   (w=%.0f%%): %s (%.1f%%)",
            weights.get("text", 0) * 100,
            text_e.get("emotion", "n/a").upper(),
            text_e.get("confidence", 0.0) * 100,
        )


def _print_qg_terminal(questions: dict) -> None:
    if not questions.get("enabled") or questions.get("error"):
        reason = questions.get("error") or "QG disabled"
        log.info("QG: %s", reason)
        return
    lat = questions.get("latency_ms", 0)
    qs = questions.get("questions", [])
    log.info("QG: %d questions in %dms", len(qs), lat)
    for i, q in enumerate(qs, 1):
        log.info("  Q%d: %s", i, q)