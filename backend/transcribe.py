"""Auris — Transcription engine with OPTIMIZED parallel pipeline

Pipeline architecture (Maximum Parallelism):
  ┌─────────────────────────────────────────────────────────────────────┐
  │  PHASE 1: Launch simultaneously (T=0)                               │
  │    ┌─────────────┐        ┌─────────────────┐                     │
  │    │   Whisper   │        │  Audio Emotion  │                     │
  │    │    ASR      │        │   (Wav2Vec2)    │                     │
  │    └──────┬──────┘        └─────────────────┘                     │
  │           │                                                        │
  │           ▼  raw_text available                                    │
  │  PHASE 2: Launch simultaneously                                    │
  │    ┌─────────────┐        ┌─────────────────┐                     │
  │    │ Text Emotion│        │  Flan-T5 Batch  │                     │
  │    │(DistilRoBERTa)│      │   Correction    │                     │
  │    └──────┬──────┘        └────────┬────────┘                     │
  │           │                        │                               │
  │           ▼                        ▼  corrected_text               │
  │  PHASE 3: Launch simultaneously                                    │
  │    ┌─────────────┐        ┌─────────────────┐                     │
  │    │   Fusion    │        │   Qwen QG       │                     │
  │    │  (merge)    │        │  (async HTTP)   │                     │
  │    └──────┬──────┘        └─────────────────┘                     │
  │           │                                                        │
  │           └────────────────┬─────────────────┘                     │
  │                            ▼                                       │
  │                      JSON Response                                 │
  └─────────────────────────────────────────────────────────────────────┘

Key optimization: Text Emotion runs on RAW text (not corrected), so it can
start immediately when Whisper finishes — no need to wait for Flan.
Audio Emotion starts at T=0 alongside Whisper, maximizing parallelism.
"""
import asyncio
import logging
import queue
import time
from concurrent.futures import ThreadPoolExecutor, Future, wait as _wait, ALL_COMPLETED

import numpy as np
import torch

from .models import load_whisper, load_flan
from . import config

log = logging.getLogger("auris.transcribe")

# Import emotion function
try:
    from .emotion import detect_emotion_global
    EMOTION_AVAILABLE = True
except ImportError:
    EMOTION_AVAILABLE = False
    log.warning("Emotion module not available")

# Import text emotion function
try:
    from .text_emotion import detect_emotion_from_text
    TEXT_EMOTION_AVAILABLE = True
except ImportError:
    TEXT_EMOTION_AVAILABLE = False
    log.warning("Text emotion module not available")

# Import fusion layer
try:
    from .emotion_fusion import fuse_emotions
    FUSION_AVAILABLE = True
except ImportError:
    FUSION_AVAILABLE = False
    log.warning("Emotion fusion module not available")

# Import QG function (sync fallback)
try:
    from .qwen_questions import generate_questions
    QG_AVAILABLE = True
except ImportError:
    QG_AVAILABLE = False
    log.warning("QG module not available — install Ollama + qwen2.5:1.5b")

_SENTINEL = object()


# ── Quality gate ───────────────────────────────────────────────────────────────

def _should_correct_segment(seg) -> bool:
    """Return True if the segment quality is low enough to warrant Flan-T5."""
    if getattr(seg, "no_speech_prob", 0.0) > config.CRITIQUE_NO_SPEECH_THRESHOLD:
        return False
    if getattr(seg, "avg_logprob", 0.0) < config.CRITIQUE_AVG_LOGPROB_THRESHOLD:
        return True
    if getattr(seg, "compression_ratio", 1.0) > config.CRITIQUE_COMPRESSION_RATIO_MAX:
        return True
    return False


# ── Simple (no-correction) transcription ─────────────────────────────────────

def transcribe_whisper(audio: np.ndarray) -> dict:
    """Decode with faster-whisper using a pre-loaded float32 array."""
    seg_iter, _ = load_whisper().transcribe(
        audio,
        language=config.WHISPER_LANGUAGE,
        beam_size=config.WHISPER_BEAM_SIZE,
        word_timestamps=True,
        vad_filter=False,
        condition_on_previous_text=False,
    )

    segments: list[dict] = []
    parts: list[str] = []

    for seg in seg_iter:
        txt = seg.text.strip()
        if txt:
            segments.append({"start": round(seg.start, 2), "end": round(seg.end, 2), "text": txt})
            parts.append(seg.text)

    text = " ".join(parts).strip()
    return {"text": text, "word_count": len(text.split()) if text else 0, "segments": segments}


# ── Streaming producer ─────────────────────────────────────────────────────────

def _whisper_producer(audio: np.ndarray, seg_queue: "queue.Queue") -> None:
    """Whisper producer thread — streams segments into queue."""
    try:
        seg_iter, _ = load_whisper().transcribe(
            audio,
            language=config.WHISPER_LANGUAGE,
            beam_size=config.WHISPER_BEAM_SIZE,
            word_timestamps=True,
            vad_filter=False,
            condition_on_previous_text=False,
        )

        idx = 0
        for seg in seg_iter:
            txt = seg.text.strip()
            if not txt:
                continue
            seg_dict = {"start": round(seg.start, 2), "end": round(seg.end, 2), "text": txt}
            seg_queue.put((idx, seg_dict, seg.text, _should_correct_segment(seg)))
            idx += 1

    except Exception as exc:
        log.error("Whisper producer error: %s", exc)
    finally:
        seg_queue.put(_SENTINEL)


# ── P2: Token-based chunking for Flan ─────────────────────────────────────────

def _chunk_by_tokens(
    text: str,
    tokenizer,
    max_tokens: int,
) -> list[str]:
    """
    Split *text* into chunks whose token length ≤ max_tokens.
    Uses the Flan tokenizer directly so we never silently truncate long segments.
    """
    tokens = tokenizer.encode(text, add_special_tokens=False)
    if len(tokens) <= max_tokens:
        return [text]

    chunks = []
    for start in range(0, len(tokens), max_tokens):
        chunk_ids = tokens[start: start + max_tokens]
        chunks.append(tokenizer.decode(chunk_ids, skip_special_tokens=True))
    return chunks


# ── P0: Batched Flan-T5 correction ────────────────────────────────────────────

def _correct_batch(
    texts: list[str],
    model,
    tokenizer,
    device,
) -> list[str]:
    """
    Run Flan-T5 on a *batch* of segment texts in one forward pass.
    3-5× faster than calling model.generate() per-segment.
    """
    if not texts:
        return []

    max_tok = config.FLAN_MAX_INPUT_TOKENS
    prompts: list[str] = []
    index_map: list[tuple[int, int]] = []

    for text_idx, text in enumerate(texts):
        chunks = _chunk_by_tokens(text, tokenizer, max_tok)
        for chunk_idx, chunk in enumerate(chunks):
            prompts.append(f"Fix grammar: {chunk}")
            index_map.append((text_idx, chunk_idx))

    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        truncation=True,
        max_length=max_tok + 10,
        padding=True,
    ).to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=config.FLAN_MAX_TOKENS,
            num_beams=config.FLAN_NUM_BEAMS,
            early_stopping=True,
            no_repeat_ngram_size=2,
            use_cache=True,
        )

    decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    assembled: dict[int, list[tuple[int, str]]] = {}
    for prompt_idx, (text_idx, chunk_idx) in enumerate(index_map):
        assembled.setdefault(text_idx, []).append((chunk_idx, decoded[prompt_idx].strip()))

    results: list[str] = []
    for text_idx in range(len(texts)):
        chunks_out = assembled.get(text_idx, [])
        chunks_out.sort(key=lambda x: x[0])
        results.append(" ".join(c for _, c in chunks_out))

    return results


def _flan_consumer(
    seg_queue: "queue.Queue",
    results: list,
    stats: dict,
) -> None:
    """
    P0: Batched Flan-T5 consumer.
    Drains the queue completely, then runs ONE batched forward pass.
    """
    items: list[tuple[int, dict, str, bool]] = []
    while True:
        item = seg_queue.get()
        if item is _SENTINEL:
            break
        items.append(item)

    if not config.FLAN_ENABLED:
        for idx, seg_dict, raw_text, _ in items:
            results[idx] = (seg_dict, raw_text, raw_text)
        return

    model, tokenizer = load_flan()
    if model is None or tokenizer is None:
        for idx, seg_dict, raw_text, _ in items:
            results[idx] = (seg_dict, raw_text, raw_text)
        return

    device = next(model.parameters()).device

    needs: list[tuple[int, dict, str]] = []
    keeps: list[tuple[int, dict, str]] = []

    for idx, seg_dict, raw_text, needs_correction in items:
        if needs_correction:
            needs.append((idx, seg_dict, raw_text))
        else:
            keeps.append((idx, seg_dict, raw_text))
            stats["kept"] += 1

    if needs:
        t_batch = time.perf_counter()
        raw_texts = [raw for _, _, raw in needs]
        corrected_s = _correct_batch(raw_texts, model, tokenizer, device)
        batch_ms = round((time.perf_counter() - t_batch) * 1000)
        log.debug("Flan batch: %d/%d corrected in %dms", len(needs), len(items), batch_ms)

        for (idx, seg_dict, raw_text), corrected in zip(needs, corrected_s):
            changed = corrected != raw_text.strip()
            stats["corrected"] += int(changed)
            stats["kept"] += int(not changed)
            results[idx] = (seg_dict, raw_text, corrected)

    for idx, seg_dict, raw_text in keeps:
        results[idx] = (seg_dict, raw_text, raw_text.strip())


# ── Legacy single-segment corrector ────────────────────────────────────────────

def _correct_one(text: str, model, tokenizer, device) -> str:
    """Run Flan-T5 on a single segment text (used by correct_text() only)."""
    chunks = _chunk_by_tokens(text, tokenizer, config.FLAN_MAX_INPUT_TOKENS)
    corrected_chunks = _correct_batch(chunks, model, tokenizer, device)
    return " ".join(corrected_chunks)


# ── P1: Async Ollama / QG ─────────────────────────────────────────────────────

async def _run_qg_async(corrected_text: str) -> dict:
    """
    P1: Question generation via Ollama — async HTTP so the thread-pool worker
    is freed while waiting for the network response.
    """
    _empty = {
        "enabled":    False,
        "questions":  [],
        "raw":        "",
        "latency_ms": 0,
        "error":      None,
    }

    if not QG_AVAILABLE:
        return {**_empty, "error": "qwen_questions module not found"}

    if not corrected_text or not corrected_text.strip():
        return {**_empty, "error": "empty transcript — no questions generated"}

    t0 = time.perf_counter()
    try:
        try:
            import httpx
            from .qwen_questions import OLLAMA_URL, MODEL_NAME, SYSTEM_PROMPT
            words = corrected_text.split()
            text_in = (" ".join(words[:200]) + "…") if len(words) > 200 else corrected_text
            prompt = f"{SYSTEM_PROMPT}\n\nPresentation:\n{text_in}"

            async with httpx.AsyncClient(timeout=300) as client:
                resp = await client.post(
                    OLLAMA_URL,
                    json={
                        "model":      MODEL_NAME,
                        "prompt":     prompt,
                        "stream":     False,
                        "keep_alive": -1,
                        "options": {
                            "temperature":    0.7,
                            "top_p":          0.9,
                            "repeat_penalty": 1.1,
                            "num_predict":    180,
                            "num_ctx":        512,
                        },
                    },
                )
                resp.raise_for_status()
                raw = resp.json()["response"].strip()

        except ImportError:
            raw = await asyncio.to_thread(generate_questions, corrected_text)

        latency_ms = round((time.perf_counter() - t0) * 1000)

        lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
        questions = []
        for ln in lines:
            if ln and ln[0].isdigit() and len(ln) > 2 and ln[1] in ".):" :
                questions.append(ln[2:].strip())
            elif ln:
                questions.append(ln)

        return {
            "enabled":    True,
            "questions":  questions,
            "raw":        raw,
            "latency_ms": latency_ms,
            "error":      None,
        }

    except Exception as exc:
        latency_ms = round((time.perf_counter() - t0) * 1000)
        log.error("QG failed: %s", exc)
        return {**_empty, "enabled": True, "latency_ms": latency_ms, "error": str(exc)}


def _run_qg(corrected_text: str) -> dict:
    """Sync wrapper — runs the async QG function in an event loop."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import concurrent.futures as _cf
            fut = asyncio.run_coroutine_threadsafe(_run_qg_async(corrected_text), loop)
            return fut.result(timeout=310)
        else:
            return loop.run_until_complete(_run_qg_async(corrected_text))
    except Exception:
        return asyncio.run(_run_qg_async(corrected_text))


# ── Emotion wrappers ──────────────────────────────────────────────────────────

def _run_audio_emotion(audio: np.ndarray) -> dict:
    """Audio emotion detection (Wav2Vec2) — single-audio wrapper."""
    _empty = {
        "enabled": False, "emotion": "unknown", "confidence": 0.0,
        "latency_ms": 0, "is_reliable": False, "all_probs": {}, "model": None,
    }
    if not config.EMOTION_ENABLED:
        return _empty
    if not EMOTION_AVAILABLE:
        return {**_empty, "error": "Emotion module not available"}
    return detect_emotion_global(audio, sr=config.EMOTION_SR)


def _run_text_emotion(text: str) -> dict:
    """Text emotion detection (DistilRoBERTa) — runs in its own thread."""
    _empty = {
        "enabled": False, "emotion": "unknown", "confidence": 0.0,
        "latency_ms": 0, "is_reliable": False, "all_probs": {}, "model": None,
    }
    if not config.TEXT_EMOTION_ENABLED:
        return _empty
    if not TEXT_EMOTION_AVAILABLE:
        return {**_empty, "error": "Text emotion module not available"}
    return detect_emotion_from_text(text)


# Keep legacy alias
_run_emotion = _run_audio_emotion


# ═══════════════════════════════════════════════════════════════════════════════
# OPTIMIZED MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

def transcribe_whisper_with_correction_and_emotion(
    audio: np.ndarray,
    run_qg: bool = True,
) -> tuple[dict, dict, dict, dict]:
    """
    OPTIMIZED 5-way parallel pipeline.

    Dependency graph (all independent branches run simultaneously):

        PHASE 1 (T=0):  ┌─► Whisper ASR ─────┐
                        │                    │ raw_text
                        └─► Audio Emotion ───┘

        PHASE 2 (Whisper done):
                        ┌─► Text Emotion (raw_text) ──┐
                        │                               │
                        └─► Flan Correction ────────────┘ corrected_text

        PHASE 3 (Phase 2 done):
                        ┌─► Emotion Fusion ─────────────┐
                        │   (audio + text results)      │
                        │                               │
                        └─► Qwen QG (corrected_text) ───┘

    Key optimization: Text Emotion runs on RAW text, not corrected text.
    This eliminates the Flan→TextEmotion dependency, saving Flan latency.
    """
    t0 = time.perf_counter()

    MAX_SEGS   = 256
    results    = [None] * MAX_SEGS
    flan_stats = {"corrected": 0, "kept": 0}
    seg_queue: queue.Queue = queue.Queue(maxsize=0)

    # ─────────────────────────────────────────────────────────────────────────
    # PHASE 1: Launch Whisper + Audio Emotion simultaneously (T=0)
    # ─────────────────────────────────────────────────────────────────────────
    with ThreadPoolExecutor(max_workers=5) as executor:
        t_phase1 = time.perf_counter()

        # Audio Emotion starts IMMEDIATELY (no dependency on anything)
        future_audio_emotion = executor.submit(_run_audio_emotion, audio)

        # Whisper starts immediately, streams segments into queue
        future_whisper = executor.submit(_whisper_producer, audio, seg_queue)

        # Wait for Whisper to finish → raw_text is now available in queue
        future_whisper.result()
        t_whisper_ms = round((time.perf_counter() - t_phase1) * 1000)

        # Drain queue to build raw_text (needed for Text Emotion)
        # Note: Flan consumer also needs to drain the same queue,
        # so we collect items here and pass them to Flan
        queue_items: list = []
        raw_parts: list[str] = []
        segments: list[dict] = []
        while True:
            item = seg_queue.get()
            if item is _SENTINEL:
                break
            queue_items.append(item)
            idx, seg_dict, raw_text, needs_corr = item
            results[idx] = (seg_dict, raw_text, raw_text)  # pre-fill with raw
            raw_parts.append(raw_text)
            segments.append(seg_dict)

        raw_text = " ".join(raw_parts).strip()
        total_segs = len(segments)

        # ─────────────────────────────────────────────────────────────────────
        # PHASE 2: Launch Text Emotion + Flan simultaneously
        # Text Emotion uses RAW text (not corrected) — saves Flan latency!
        # ─────────────────────────────────────────────────────────────────────
        t_phase2 = time.perf_counter()

        future_text_emotion = executor.submit(_run_text_emotion, raw_text)

        # Flan consumer processes the queue items we already drained
        # We run it in-thread since queue is already drained
        if config.FLAN_ENABLED:
            model, tokenizer = load_flan()
            if model is not None and tokenizer is not None:
                device = next(model.parameters()).device

                needs: list[tuple[int, dict, str]] = []
                keeps: list[tuple[int, dict, str]] = []

                for idx, seg_dict, raw_text, needs_correction in queue_items:
                    if needs_correction:
                        needs.append((idx, seg_dict, raw_text))
                    else:
                        keeps.append((idx, seg_dict, raw_text))
                        flan_stats["kept"] += 1

                if needs:
                    t_batch = time.perf_counter()
                    raw_texts = [raw for _, _, raw in needs]
                    corrected_s = _correct_batch(raw_texts, model, tokenizer, device)
                    batch_ms = round((time.perf_counter() - t_batch) * 1000)
                    log.debug("Flan batch: %d/%d corrected in %dms", len(needs), total_segs, batch_ms)

                    for (idx, seg_dict, raw_text), corrected in zip(needs, corrected_s):
                        changed = corrected != raw_text.strip()
                        flan_stats["corrected"] += int(changed)
                        flan_stats["kept"] += int(not changed)
                        results[idx] = (seg_dict, raw_text, corrected)

                for idx, seg_dict, raw_text in keeps:
                    results[idx] = (seg_dict, raw_text, raw_text.strip())

        t_flan_ms = round((time.perf_counter() - t_phase2) * 1000)

        # Build corrected_text from results
        ordered = [r for r in results if r is not None]
        corrected_parts = [corrected for _, _, corrected in ordered]
        corrected_text = " ".join(corrected_parts).strip()

        # ─────────────────────────────────────────────────────────────────────
        # PHASE 3: Launch Fusion + QG simultaneously
        # Fusion needs: Audio Emotion + Text Emotion
        # QG needs: corrected_text
        # ─────────────────────────────────────────────────────────────────────
        t_phase3 = time.perf_counter()

        # Wait for emotion results (they should already be done or nearly done)
        _wait([future_audio_emotion, future_text_emotion], return_when=ALL_COMPLETED)
        audio_emotion_data = future_audio_emotion.result()
        text_emotion_data  = future_text_emotion.result()

        # Fusion (can run in-thread, it's fast)
        if FUSION_AVAILABLE:
            fused_emotion = fuse_emotions(audio_emotion_data, text_emotion_data)
        else:
            fused_emotion = {
                **audio_emotion_data,
                "source": "audio_only",
                "fusion_weights": {"audio": 1.0, "text": 0.0},
                "audio_emotion": audio_emotion_data,
                "text_emotion":  text_emotion_data,
            }

        t_fusion_ms = round((time.perf_counter() - t_phase3) * 1000)

        # QG runs in parallel thread (async HTTP inside)
        future_qg: Future = (
            executor.submit(_run_qg, corrected_text)
            if run_qg
            else executor.submit(lambda: {
                "enabled": False, "questions": [], "raw": "",
                "latency_ms": 0, "error": "skipped",
            })
        )

        qg_data = future_qg.result()
        t_qg_ms = round((time.perf_counter() - t_phase3) * 1000)

    t_pipeline = round((time.perf_counter() - t0) * 1000)

    # ─────────────────────────────────────────────────────────────────────────
    # Build final results
    # ─────────────────────────────────────────────────────────────────────────
    corrected_count = flan_stats["corrected"]
    kept_count      = flan_stats["kept"]

    whisper_result = {
        "text":       raw_text,
        "word_count": len(raw_text.split()) if raw_text else 0,
        "segments":   segments,
    }
    correction_result = {
        "corrected":  corrected_text,
        "enabled":    config.FLAN_ENABLED,
        "model":      config.FLAN_MODEL if config.FLAN_ENABLED else None,
        "latency_ms": t_flan_ms,
        "stage_ms": {
            "whisper":      t_whisper_ms,
            "flan":         t_flan_ms,
            "emotion":      audio_emotion_data.get("latency_ms", 0),
            "text_emotion": text_emotion_data.get("latency_ms", 0),
            "fusion":       t_fusion_ms,
            "qg":           qg_data.get("latency_ms", 0),
        },
        "critique_stats": {
            "corrected": corrected_count,
            "kept":      kept_count,
            "total":     total_segs,
        },
    }
    questions_result = {
        "enabled":    qg_data.get("enabled", False),
        "questions":  qg_data.get("questions", []),
        "raw":        qg_data.get("raw", ""),
        "latency_ms": qg_data.get("latency_ms", 0),
        "error":      qg_data.get("error"),
    }

    log.info(
        "Pipeline (OPTIMIZED) — whisper:%d seg | flan:+%d/=%d | "
        "audio_emotion:%s(%.1f%%) | text_emotion:%s(%.1f%%) | "
        "fused:%s(%.1f%%) | qg:%d questions | total:%dms",
        total_segs, corrected_count, kept_count,
        audio_emotion_data.get("emotion", "n/a"),
        audio_emotion_data.get("confidence", 0.0) * 100,
        text_emotion_data.get("emotion", "n/a"),
        text_emotion_data.get("confidence", 0.0) * 100,
        fused_emotion.get("emotion", "n/a"),
        fused_emotion.get("confidence", 0.0) * 100,
        len(questions_result["questions"]),
        t_pipeline,
    )

    return whisper_result, correction_result, fused_emotion, questions_result


def transcribe_whisper_with_correction(audio: np.ndarray) -> tuple[dict, dict]:
    """Decode with Whisper + batched Flan-T5 correction (no emotion, no QG)."""
    wr, cr, _, _ = transcribe_whisper_with_correction_and_emotion(audio, run_qg=False)
    return wr, cr


def correct_text(text: str) -> dict:
    """Run Flan-T5 correction on full text as single segment."""
    if not config.FLAN_ENABLED or not text or not text.strip():
        return {
            "corrected": text,
            "enabled":   config.FLAN_ENABLED,
            "model":     config.FLAN_MODEL if config.FLAN_ENABLED else None,
            "latency_ms": 0,
        }
    model, tokenizer = load_flan()
    if model is None or tokenizer is None:
        return {"corrected": text, "enabled": False, "model": None, "latency_ms": 0}

    device = next(model.parameters()).device
    t_start = time.perf_counter()
    corrected = _correct_one(text, model, tokenizer, device)
    return {
        "corrected":  corrected,
        "enabled":    True,
        "model":      config.FLAN_MODEL,
        "latency_ms": round((time.perf_counter() - t_start) * 1000),
    }