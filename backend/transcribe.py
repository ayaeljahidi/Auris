"""Auris — Transcription engine with streaming Flan-T5 correction

Pipeline architecture (3-way parallel):
  ┌─────────────┐   Queue   ┌─────────────┐
  │   Whisper   │ ─────────▶│   Flan-T5   │   (overlap: Flan starts on seg-1
  │  (iterator) │           │  consumer   │    while Whisper decodes seg-2…N)
  └─────────────┘           └─────────────┘
         │                                          Both run concurrently with:
         └──────────────────────────────────────▶  ┌──────────────┐
                                                    │   Emotion    │
                                                    │  (parallel)  │
                                                    └──────────────┘

Key change vs previous version:
  BEFORE: Whisper runs fully → then Flan runs on all segments (sequential phases)
  AFTER:  Whisper streams segments into a queue; Flan corrects each one immediately
          → Flan latency is hidden inside Whisper decode time.
          For 17 segments the savings are roughly (N-1) × flan_per_segment_ms.
"""
import logging
import queue
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch

from .models import load_whisper, load_flan
from . import config

log = logging.getLogger("auris.transcribe")

# Import emotion function (no circular import)
try:
    from .emotion import detect_emotion_global
    EMOTION_AVAILABLE = True
except ImportError:
    EMOTION_AVAILABLE = False
    log.warning("Emotion module not available")

# Sentinel pushed by Whisper thread to signal end-of-stream to Flan consumer
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

    Each item pushed: (index, seg_dict, raw_text, needs_correction: bool)
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
    """Run Flan-T5 on a single segment text. Returns corrected text."""
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
    results: list,          # pre-sized list, written by index
    stats: dict,            # mutated in-place: corrected / kept counts
) -> None:
    """
    Flan-T5 consumer thread.
    Pops segments from the queue and corrects them immediately.
    Runs concurrently with _whisper_producer so Flan latency is overlapped.
    """
    if not config.FLAN_ENABLED:
        # Drain queue without correction; results filled by _collect_uncorrected
        while True:
            item = seg_queue.get()
            if item is _SENTINEL:
                break
            idx, seg_dict, raw_text, _ = item
            results[idx] = (seg_dict, raw_text, raw_text)  # (seg, raw, corrected)
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


# ── Emotion helper (unchanged interface) ──────────────────────────────────────

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


# ── Batch correction (kept for _batch_correct_segments callers) ───────────────

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


# ── Main pipeline ──────────────────────────────────────────────────────────────

def transcribe_whisper_with_correction_and_emotion(audio: np.ndarray) -> tuple[dict, dict, dict]:
    """
    Three-way parallel pipeline:

      Thread-1  Whisper       — streams segments into a queue as decoded
      Thread-2  Flan-T5       — consumes queue, corrects each segment immediately
      Thread-3  Emotion ONNX  — runs on full audio independently

    Flan-T5 now overlaps with Whisper decode instead of running after it.
    For 17 segments at ~200ms/segment Flan latency this saves ~3 seconds.

    Timeline (old):
      [=====Whisper=====][===Flan batch===][Emotion]
                                                     ^ total

    Timeline (new):
      [=====Whisper======]
              [=Flan seg1=][=seg2=][=seg3=]…   (starts on seg-1 arrival)
      [==========Emotion===========]
                                   ^ total  (Flan hidden inside Whisper)
    """
    t0 = time.perf_counter()

    # We don't know the segment count upfront; use a growable list protected
    # by the queue ordering.  We pre-allocate 256 slots (cheap) and trim later.
    MAX_SEGS = 256
    results: list = [None] * MAX_SEGS   # results[idx] = (seg_dict, raw, corrected)
    flan_stats = {"corrected": 0, "kept": 0}

    # Bounded queue: Whisper can run 2 segments ahead of Flan (back-pressure)
    seg_queue: queue.Queue = queue.Queue(maxsize=4)

    with ThreadPoolExecutor(max_workers=3) as executor:
        # Thread 1: Whisper producer
        future_whisper = executor.submit(_whisper_producer, audio, seg_queue)
        # Thread 2: Flan consumer (starts immediately, blocks on queue.get)
        future_flan    = executor.submit(_flan_consumer, seg_queue, results, flan_stats)
        # Thread 3: Emotion (full audio, independent)
        future_emotion = executor.submit(_run_emotion, audio)

        # Wait for both Whisper and Flan to finish
        future_whisper.result()
        future_flan.result()
        emotion_data = future_emotion.result()

    t_pipeline = round((time.perf_counter() - t0) * 1000)

    # Collect ordered results (trim None slots)
    ordered = [r for r in results if r is not None]

    segments:        list[dict] = []
    raw_parts:       list[str]  = []
    corrected_parts: list[str]  = []

    for seg_dict, raw, corrected in ordered:
        segments.append(seg_dict)
        raw_parts.append(raw)
        corrected_parts.append(corrected)

    text           = " ".join(raw_parts).strip()
    corrected_text = " ".join(corrected_parts).strip()
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
        "enabled":        emotion_data.get("enabled", config.EMOTION_ENABLED),
        "emotion":        emotion_data.get("emotion", "unknown"),
        "confidence":     emotion_data.get("confidence", 0.0),
        "latency_ms":     emotion_data.get("latency_ms", 0),
        "is_reliable":    emotion_data.get("is_reliable", False),
        "all_probs":      emotion_data.get("all_probs", {}),
        "realtime_factor": emotion_data.get("realtime_factor", 0),
        "inference_ms":   emotion_data.get("inference_ms", 0),
    }

    log.info(
        "✅ Pipeline — whisper:%d seg | flan:+%d/=%d | emotion:%s(%.1f%%) | total:%dms | %.1fx realtime",
        total_segs, corrected_count, kept_count,
        emotion_result["emotion"], emotion_result["confidence"] * 100,
        t_pipeline, emotion_result.get("realtime_factor", 0),
    )

    return whisper_result, correction_result, emotion_result


# ── Convenience wrappers (unchanged signatures) ───────────────────────────────

def transcribe_whisper_with_correction(audio: np.ndarray) -> tuple[dict, dict]:
    """Decode with Whisper + streaming Flan-T5 correction."""
    wr, cr, _ = transcribe_whisper_with_correction_and_emotion(audio)
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
    