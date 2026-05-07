"""Auris — Transcription engine with streaming Flan-T5 correction + QG

Pipeline architecture (4-way parallel):
  ┌─────────────┐   Queue   ┌─────────────┐
  │   Whisper   │ ─────────▶│   Flan-T5   │   (overlap: Flan starts on seg-1
  │  (iterator) │           │  consumer   │    while Whisper decodes seg-2…N)
  └─────────────┘           └─────────────┘
         │                        │
         │                        └─── corrected_text ──▶ ┌──────────────┐
         │                                                  │  Qwen QG     │
         │                                                  │  (Ollama)    │
         └──────────────────────────────────────────────▶  └──────────────┘
                                                    AND:
                                                    ┌──────────────┐
                                                    │   Emotion    │
                                                    │  (parallel)  │
                                                    └──────────────┘

Timeline:
  [=====Whisper======]
          [=Flan s1=][=s2=][=s3=]…  ← Flan overlaps Whisper
  [==========Emotion===========]    ← fully independent
                                [==Qwen QG==]  ← starts after Flan completes,
                                                  overlaps with tail of Emotion
"""
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

# Import QG function
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


# ── Simple (no-correction) transcription ──────────────────────────────────────

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


# ── Streaming producer-consumer helpers ───────────────────────────────────────

def _whisper_producer(audio: np.ndarray, seg_queue: "queue.Queue") -> None:
    """
    Whisper producer thread.
    Streams segments into seg_queue as soon as each one is decoded.
    Pushes _SENTINEL when done.
    Each item: (index, seg_dict, raw_text, needs_correction: bool)
    """
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


def _correct_one(text: str, model, tokenizer, device) -> str:
    """Run Flan-T5 on a single segment text."""
    inputs = tokenizer(
        f"Fix grammar: {text}",
        return_tensors="pt",
        truncation=True,
        max_length=256,
    ).to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=config.FLAN_MAX_TOKENS,
            num_beams=config.FLAN_NUM_BEAMS,
            early_stopping=True,
            no_repeat_ngram_size=2,
        )

    return tokenizer.decode(outputs[0], skip_special_tokens=True).strip()


def _flan_consumer(
    seg_queue: "queue.Queue",
    results: list,
    stats: dict,
) -> None:
    """
    Flan-T5 consumer thread.
    Pops segments from the queue and corrects them immediately.
    Runs concurrently with _whisper_producer.
    """
    if not config.FLAN_ENABLED:
        while True:
            item = seg_queue.get()
            if item is _SENTINEL:
                break
            idx, seg_dict, raw_text, _ = item
            results[idx] = (seg_dict, raw_text, raw_text)
        return

    model, tokenizer = load_flan()
    if model is None or tokenizer is None:
        while True:
            item = seg_queue.get()
            if item is _SENTINEL:
                break
            idx, seg_dict, raw_text, _ = item
            results[idx] = (seg_dict, raw_text, raw_text)
        return

    device = next(model.parameters()).device

    while True:
        item = seg_queue.get()
        if item is _SENTINEL:
            break

        idx, seg_dict, raw_text, needs_correction = item

        if needs_correction:
            t_seg = time.perf_counter()
            corrected = _correct_one(raw_text, model, tokenizer, device)
            seg_ms = round((time.perf_counter() - t_seg) * 1000)
            changed = corrected != raw_text.strip()
            stats["corrected"] += int(changed)
            stats["kept"] += int(not changed)
            log.debug(
                "Flan seg[%d] %dms — %s→ %s",
                idx, seg_ms,
                "✏  " if changed else "✓  ",
                corrected[:60],
            )
        else:
            corrected = raw_text.strip()
            stats["kept"] += 1

        results[idx] = (seg_dict, raw_text, corrected)


# ── Emotion helper ─────────────────────────────────────────────────────────────

def _run_emotion(audio: np.ndarray) -> dict:
    """Emotion detection — runs in its own thread."""
    if not config.EMOTION_ENABLED:
        return {"enabled": False, "emotion": "unknown", "confidence": 0.0,
                "latency_ms": 0, "is_reliable": False, "all_probs": {}}
    if not EMOTION_AVAILABLE:
        return {"enabled": False, "emotion": "unknown", "confidence": 0.0,
                "latency_ms": 0, "is_reliable": False, "all_probs": {},
                "error": "Emotion module not available"}
    return detect_emotion_global(audio, sr=config.EMOTION_SR)


# ── QG helper ─────────────────────────────────────────────────────────────────

def _run_qg(corrected_text: str) -> dict:
    """
    Question generation via Qwen/Ollama.
    Runs AFTER Flan completes (needs full corrected transcript).
    Returns a questions_result dict — always safe to call, returns
    a graceful failure dict if Ollama is not available.
    """
    _empty = {
        "enabled": False,
        "questions": [],
        "raw": "",
        "latency_ms": 0,
        "error": None,
    }

    if not QG_AVAILABLE:
        return {**_empty, "error": "qwen_questions module not found"}

    if not corrected_text or not corrected_text.strip():
        return {**_empty, "error": "empty transcript — no questions generated"}

    t0 = time.perf_counter()
    try:
        raw = generate_questions(corrected_text)
        latency_ms = round((time.perf_counter() - t0) * 1000)

        # Parse numbered lines into a clean list
        lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
        questions = []
        for ln in lines:
            # Strip leading "1." / "2." etc.
            if ln and ln[0].isdigit() and len(ln) > 2 and ln[1] in ".):":
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
            "enabled":    True,   # was attempted
            "latency_ms": latency_ms,
            "error":      str(exc),
        }


# ── Batch correction (kept for external callers) ──────────────────────────────

def _batch_correct_segments(texts: list[str]) -> list[str]:
    """Batch Flan-T5 — kept for external callers; streaming path is preferred."""
    if not texts:
        return []
    model, tokenizer = load_flan()
    if model is None or tokenizer is None:
        return texts
    device = next(model.parameters()).device
    prompts = [f"Fix grammar: {t}" for t in texts]
    inputs = tokenizer(
        prompts, return_tensors="pt", padding=True, truncation=True, max_length=256,
    ).to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=config.FLAN_MAX_TOKENS,
            num_beams=config.FLAN_NUM_BEAMS,
            early_stopping=True,
            no_repeat_ngram_size=2,
        )
    return [tokenizer.decode(out, skip_special_tokens=True).strip() for out in outputs]


# ── Main 4-way pipeline ────────────────────────────────────────────────────────

def transcribe_whisper_with_correction_and_emotion(
    audio: np.ndarray,
    run_qg: bool = True,
) -> tuple[dict, dict, dict, dict]:
    """
    Four-way pipeline:

      Thread-1  Whisper       — streams segments into a bounded queue
      Thread-2  Flan-T5       — consumes queue, corrects each segment immediately
      Thread-3  Emotion ONNX  — runs on full audio independently (full parallel)

      After Threads 1+2 complete:
      Thread-4  Qwen QG       — launched with corrected_text, may overlap
                                 with Thread-3 tail if Emotion is still running

    Timeline:
      [=====Whisper======]
              [=Flan s1=][=s2=]…
      [==========Emotion===========]
                                [==Qwen QG==]   ← starts as soon as Flan done

    Args:
        audio:   float32 numpy array, mono 16 kHz
        run_qg:  set False to skip QG (e.g. live/real-time mode)

    Returns:
        (whisper_result, correction_result, emotion_result, questions_result)
    """
    t0 = time.perf_counter()

    MAX_SEGS   = 256
    results    = [None] * MAX_SEGS
    flan_stats = {"corrected": 0, "kept": 0}
    seg_queue: queue.Queue = queue.Queue(maxsize=4)

    with ThreadPoolExecutor(max_workers=4) as executor:
        # Thread 1 + 2: Whisper→queue→Flan (streaming overlap)
        future_whisper = executor.submit(_whisper_producer, audio, seg_queue)
        future_flan    = executor.submit(_flan_consumer, seg_queue, results, flan_stats)
        # Thread 3: Emotion — fully independent, runs the whole time
        future_emotion = executor.submit(_run_emotion, audio)

        # Wait for Whisper + Flan to finish before launching QG
        future_whisper.result()
        future_flan.result()

        # Assemble corrected text (needed by QG)
        ordered_partial = [r for r in results if r is not None]
        corrected_parts = [corrected for _, _, corrected in ordered_partial]
        corrected_text  = " ".join(corrected_parts).strip()

        # Thread 4: QG — starts now, may overlap with Emotion tail
        future_qg: Future = (
            executor.submit(_run_qg, corrected_text)
            if run_qg
            else executor.submit(lambda: {
                "enabled": False, "questions": [], "raw": "",
                "latency_ms": 0, "error": "skipped (live mode)",
            })
        )

        # Collect remaining results
        emotion_data = future_emotion.result()
        qg_data      = future_qg.result()

    t_pipeline = round((time.perf_counter() - t0) * 1000)

    # ── Assemble final results ─────────────────────────────────────────────────
    ordered = [r for r in results if r is not None]

    segments:        list[dict] = []
    raw_parts:       list[str]  = []
    corrected_parts2: list[str] = []

    for seg_dict, raw, corrected in ordered:
        segments.append(seg_dict)
        raw_parts.append(raw)
        corrected_parts2.append(corrected)

    text           = " ".join(raw_parts).strip()
    corrected_text = " ".join(corrected_parts2).strip()
    total_segs     = len(segments)
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
        "critique_stats": {
            "corrected": corrected_count,
            "kept":      kept_count,
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
        "✅ Pipeline — whisper:%d seg | flan:+%d/=%d | emotion:%s(%.1f%%) "
        "| qg:%d questions | total:%dms | %.1fx realtime",
        total_segs, corrected_count, kept_count,
        emotion_result["emotion"], emotion_result["confidence"] * 100,
        len(questions_result["questions"]),
        t_pipeline, emotion_result.get("realtime_factor", 0),
    )

    return whisper_result, correction_result, emotion_result, questions_result


# ── Convenience wrappers (backwards-compatible signatures) ────────────────────

def transcribe_whisper_with_correction(audio: np.ndarray) -> tuple[dict, dict]:
    """Decode with Whisper + streaming Flan-T5 correction (no emotion, no QG)."""
    wr, cr, _, _ = transcribe_whisper_with_correction_and_emotion(audio, run_qg=False)
    return wr, cr


def correct_text(text: str) -> dict:
    """Run Flan-T5 correction on full text as single segment."""
    if not config.FLAN_ENABLED or not text or not text.strip():
        return {
            "corrected": text,
            "enabled": config.FLAN_ENABLED,
            "model": config.FLAN_MODEL if config.FLAN_ENABLED else None,
            "latency_ms": 0,
        }
    model, tokenizer = load_flan()
    if model is None or tokenizer is None:
        return {"corrected": text, "enabled": False, "model": None, "latency_ms": 0}

    device = next(model.parameters()).device
    t_start = time.perf_counter()
    inputs = tokenizer(
        f"Fix grammar: {text}", return_tensors="pt", truncation=True, max_length=256,
    ).to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=config.FLAN_MAX_TOKENS,
            num_beams=config.FLAN_NUM_BEAMS,
            early_stopping=True,
            no_repeat_ngram_size=2,
        )
    corrected = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    return {
        "corrected":  corrected,
        "enabled":    True,
        "model":      config.FLAN_MODEL,
        "latency_ms": round((time.perf_counter() - t_start) * 1000),
    }