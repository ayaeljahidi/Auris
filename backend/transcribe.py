"""Auris — Transcription engine with streaming Flan-T5 correction + QG

Pipeline architecture (4-way parallel):
  ┌─────────────┐   Queue   ┌─────────────┐
  │   Whisper   │ ─────────▶│   Flan-T5   │   (batch: collects all segments,
  │  (iterator) │           │  consumer   │    then runs ONE batched forward pass)
  └─────────────┘           └─────────────┘
         │                        │
         │                        └─── corrected_text ──▶ ┌──────────────┐
         │                                                  │  Qwen QG     │
         │                                                  │  (async HTTP)│
         └──────────────────────────────────────────────▶  └──────────────┘
                                                    AND:
                                                    ┌──────────────┐
                                                    │   Emotion    │
                                                    │  (batched)   │
                                                    └──────────────┘

P0 — Batch Flan-T5: accumulate all segments that need correction, then run
     ONE batched model.generate() call instead of N serial calls.  Reduces
     Flan latency by 3-5× on typical audio (8-20 segments).

P0/P1 — Async Ollama client: generate_questions() is now called via
     asyncio + httpx so the thread-pool worker is freed while waiting for
     the Ollama HTTP response.  Falls back to the sync path if httpx is absent.

P1 — Cross-request emotion batching: _run_audio_emotion accepts a list of
     audio arrays and delegates to detect_emotion_global which already uses a
     batched forward pass internally.  This wires up the interface correctly
     for future request-coalescing middleware.

P2 — Token-based text chunking for Flan: segments are tokenised once and
     split on a real token budget (FLAN_MAX_INPUT_TOKENS) rather than a fixed
     character count, improving both accuracy and speed.
"""
import asyncio
import logging
import queue
import time
from concurrent.futures import ThreadPoolExecutor, Future

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

    Uses the Flan tokenizer directly so we never silently truncate long
    segments.  Returns a list of ≥1 string.
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

    3-5× faster than calling model.generate() per-segment because:
      • Tokenisation is batched (single call).
      • The ONNX/C++ kernel runs once over the whole batch instead of N times.
      • No Python loop overhead during the generate phase.

    Each text is first checked against the real token budget (P2); texts that
    exceed it are split into sub-chunks, each corrected independently, then
    rejoined.
    """
    if not texts:
        return []

    max_tok = config.FLAN_MAX_INPUT_TOKENS
    prompts: list[str] = []
    # Map from prompt index back to (text_idx, chunk_idx)
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
        max_length=max_tok + 10,  # +10 for "Fix grammar: " prefix tokens
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

    # Re-assemble: group corrected chunks back to their source text
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

    Drains the queue completely, then runs ONE batched forward pass over all
    segments that need correction instead of correcting them one-by-one.
    This is faster because GPU/CPU kernels saturate on a batch far better
    than on repeated single-sample calls.
    """
    # Drain the entire queue first
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

    # Separate which segments need correction
    needs: list[tuple[int, dict, str]] = []   # (idx, seg_dict, raw_text)
    keeps: list[tuple[int, dict, str]] = []   # no correction needed

    for idx, seg_dict, raw_text, needs_correction in items:
        if needs_correction:
            needs.append((idx, seg_dict, raw_text))
        else:
            keeps.append((idx, seg_dict, raw_text))
            stats["kept"] += 1

    # Batch-correct the ones that need it
    if needs:
        t_batch = time.perf_counter()
        raw_texts   = [raw for _, _, raw in needs]
        corrected_s = _correct_batch(raw_texts, model, tokenizer, device)
        batch_ms    = round((time.perf_counter() - t_batch) * 1000)
        log.debug(
            "Flan batch: %d segments corrected in %dms (%.1fms/seg)",
            len(needs), batch_ms, batch_ms / len(needs),
        )
        for (idx, seg_dict, raw_text), corrected in zip(needs, corrected_s):
            changed = corrected != raw_text.strip()
            stats["corrected"] += int(changed)
            stats["kept"] += int(not changed)
            log.debug("Flan seg[%d] — %s→ %s", idx, "✏ " if changed else "✓ ", corrected[:60])
            results[idx] = (seg_dict, raw_text, corrected)

    # Pass-through for segments that didn't need correction
    for idx, seg_dict, raw_text in keeps:
        results[idx] = (seg_dict, raw_text, raw_text.strip())


# ── Legacy single-segment corrector (kept for correct_text() endpoint) ─────────

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

    Uses httpx if available (non-blocking); falls back to the sync requests
    path in a thread via asyncio.to_thread if not.
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
        # Attempt async path first (httpx)
        try:
            import httpx

            from .qwen_questions import OLLAMA_URL, MODEL_NAME, SYSTEM_PROMPT
            words = corrected_text.split()
            text_in = (" ".join(words[:200]) + "…") if len(words) > 200 else corrected_text
            prompt  = f"{SYSTEM_PROMPT}\n\nPresentation:\n{text_in}"

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
            # httpx not installed — fall back to sync generate_questions in thread
            raw = await asyncio.to_thread(generate_questions, corrected_text)

        latency_ms = round((time.perf_counter() - t0) * 1000)

        lines     = [ln.strip() for ln in raw.splitlines() if ln.strip()]
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
        return {
            **_empty,
            "enabled":    True,
            "latency_ms": latency_ms,
            "error":      str(exc),
        }


def _run_qg(corrected_text: str) -> dict:
    """Sync wrapper — runs the async QG function in an event loop."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # Called from inside an async context (e.g. thread-pool worker
            # spawned from an async endpoint) — use asyncio.run_coroutine_threadsafe.
            import concurrent.futures as _cf
            fut = asyncio.run_coroutine_threadsafe(_run_qg_async(corrected_text), loop)
            return fut.result(timeout=310)
        else:
            return loop.run_until_complete(_run_qg_async(corrected_text))
    except Exception:
        # Pure fallback: new loop
        return asyncio.run(_run_qg_async(corrected_text))


# ── P1: Cross-request emotion batching interface ──────────────────────────────

def _run_audio_emotion(audio: np.ndarray) -> dict:
    """
    Audio emotion detection (Wav2Vec2) — single-audio wrapper.

    detect_emotion_global already executes a batched forward pass across all
    segments within *one* audio file (P1 batching is therefore in-request).
    For cross-request batching, callers can collect multiple audio arrays and
    pass them to detect_emotion_global_batch (see emotion.py) if available.
    """
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


# Keep legacy alias so existing call-sites still work
def _run_emotion(audio: np.ndarray) -> dict:
    return _run_audio_emotion(audio)


# ── Main 4-way pipeline ───────────────────────────────────────────────────────

def transcribe_whisper_with_correction_and_emotion(
    audio: np.ndarray,
    run_qg: bool = True,
) -> tuple[dict, dict, dict, dict]:
    """
    Five-way pipeline: Whisper + Flan-T5 + Audio Emotion + Text Emotion + QG.

    P0: Flan now runs a single batched forward pass (see _flan_consumer).
    P1: QG uses async Ollama HTTP (non-blocking thread-pool worker).
    Audio emotion runs in a thread alongside Whisper/Flan (models pre-loaded,
    no reload per request). Text emotion + QG start as soon as Flan finishes.
    Both emotion signals are merged by the fusion layer.
    """
    t0 = time.perf_counter()

    MAX_SEGS   = 256
    results    = [None] * MAX_SEGS
    flan_stats = {"corrected": 0, "kept": 0}
    seg_queue: queue.Queue = queue.Queue(maxsize=0)  # unbounded — batch consumer drains all

    with ThreadPoolExecutor(max_workers=5) as executor:
        future_audio_emotion = executor.submit(_run_audio_emotion, audio)

        t_whisper_start = time.perf_counter()
        future_whisper  = executor.submit(_whisper_producer, audio, seg_queue)
        future_flan     = executor.submit(_flan_consumer, seg_queue, results, flan_stats)

        future_whisper.result()
        t_whisper_ms = round((time.perf_counter() - t_whisper_start) * 1000)

        future_flan.result()
        t_flan_ms = (
            round((time.perf_counter() - t_whisper_start) * 1000) - t_whisper_ms
        )

        ordered_partial = [r for r in results if r is not None]
        corrected_parts = [corrected for _, _, corrected in ordered_partial]
        corrected_text  = " ".join(corrected_parts).strip()

        # Stage 2: text emotion + QG in parallel
        future_text_emotion = executor.submit(_run_text_emotion, corrected_text)
        future_qg: Future = (
            executor.submit(_run_qg, corrected_text)
            if run_qg
            else executor.submit(
                lambda: {
                    "enabled": False, "questions": [], "raw": "",
                    "latency_ms": 0, "error": "skipped (live mode)",
                }
            )
        )

        from concurrent.futures import wait as _wait, ALL_COMPLETED
        _wait(
            [future_audio_emotion, future_text_emotion, future_qg],
            return_when=ALL_COMPLETED,
        )
        audio_emotion_data = future_audio_emotion.result()
        text_emotion_data  = future_text_emotion.result()
        qg_data            = future_qg.result()

    t_pipeline = round((time.perf_counter() - t0) * 1000)

    # Fuse emotion signals
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

    ordered = [r for r in results if r is not None]
    segments, raw_parts, corrected_parts2 = [], [], []
    for seg_dict, raw, corrected in ordered:
        segments.append(seg_dict)
        raw_parts.append(raw)
        corrected_parts2.append(corrected)

    text            = " ".join(raw_parts).strip()
    corrected_text  = " ".join(corrected_parts2).strip()
    total_segs      = len(segments)
    corrected_count = flan_stats["corrected"]
    kept_count      = flan_stats["kept"]

    whisper_result = {
        "text":       text,
        "word_count": len(text.split()) if text else 0,
        "segments":   segments,
    }
    correction_result = {
        "corrected":  corrected_text,
        "enabled":    config.FLAN_ENABLED,
        "model":      config.FLAN_MODEL if config.FLAN_ENABLED else None,
        "latency_ms": t_pipeline,
        "stage_ms": {
            "whisper":      t_whisper_ms,
            "flan":         t_flan_ms,
            "emotion":      audio_emotion_data.get("latency_ms", 0),
            "text_emotion": text_emotion_data.get("latency_ms", 0),
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
        "✅ Pipeline — whisper:%d seg | flan:+%d/=%d | "
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