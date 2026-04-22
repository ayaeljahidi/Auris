"""Auris — Transcription engine (faster-whisper + Flan-T5)"""
import io
import logging
import time
import wave
from concurrent.futures import Future, ThreadPoolExecutor

import numpy as np

from .models import load_whisper, load_flan
from . import config

log = logging.getLogger("auris.transcribe")


# ── WAV bytes → float32 numpy array (in-memory, no disk I/O) ──────────────────

def _wav_bytes_to_float32(wav_bytes: bytes) -> np.ndarray:
    """
    Convert WAV bytes to a normalised float32 numpy array in-memory.
    Eliminates all temp-file disk I/O for Whisper.
    """
    with wave.open(io.BytesIO(wav_bytes), "rb") as w:
        pcm = w.readframes(w.getnframes())
    return np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0


# ── faster-whisper (in-memory) ─────────────────────────────────────────────────

def transcribe_whisper(wav_bytes: bytes) -> dict:
    """
    Decode WAV with faster-whisper using an in-memory float32 array.

    Returns:
        {text, word_count, segments: [{start, end, text}]}
    """
    audio = _wav_bytes_to_float32(wav_bytes)

    seg_iter, _ = load_whisper().transcribe(
        audio,
        language=config.WHISPER_LANGUAGE,
        beam_size=config.WHISPER_BEAM_SIZE,
        word_timestamps=True,
        vad_filter=False,
        condition_on_previous_text=False,
    )
    segments: list[dict] = []
    parts:    list[str]  = []

    for seg in seg_iter:
        segments.append({
            "start": round(seg.start, 2),
            "end":   round(seg.end,   2),
            "text":  seg.text.strip(),
        })
        parts.append(seg.text)

    text = " ".join(parts).strip()
    return {
        "text":       text,
        "word_count": len(text.split()) if text else 0,
        "segments":   segments,
    }


# ── Flan-T5 single-segment corrector (used inside the streaming pipeline) ──────

def _correct_segment(text: str) -> str:
    """
    Run Flan-T5 batched correction on a single segment's text.
    All sentences within the segment are batched into one generate() call.
    Returns the corrected text string (or the original on failure).
    """
    import torch

    if not text or not text.strip():
        return text

    model, tokenizer = load_flan()
    if model is None or tokenizer is None:
        return text

    device    = next(model.parameters()).device
    sentences = [s.strip() for s in text.split(".") if s.strip()]
    if not sentences:
        return text

    prompts = [
        f"Rewrite this with correct grammar and spelling: {s}"
        for s in sentences
    ]

    # Single batched tokenize + generate call
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
    ).to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=config.FLAN_MAX_TOKENS,
            num_beams=config.FLAN_NUM_BEAMS,
            early_stopping=True,
            no_repeat_ngram_size=3,
        )

    corrected_parts = [
        tokenizer.decode(out, skip_special_tokens=True).rstrip(".").strip()
        for out in outputs
    ]
    return ". ".join(corrected_parts) + "."


# ── Whisper + Flan-T5 segment-level streaming parallelism ─────────────────────

def transcribe_whisper_with_correction(wav_bytes: bytes) -> tuple[dict, dict]:
    """
    Decode WAV with faster-whisper (in-memory) while streaming each yielded
    segment immediately to Flan-T5 for correction in a background thread via
    ThreadPoolExecutor.  Whisper continues decoding the next segment
    concurrently.  Futures are collected in order and reassembled once done.

    Returns:
        (whisper_result, correction_result)
        whisper_result    — {text, word_count, segments}
        correction_result — {corrected, enabled, model, latency_ms}
    """
    if not config.FLAN_ENABLED:
        wr = transcribe_whisper(wav_bytes)
        return wr, {
            "corrected":  wr["text"],
            "enabled":    False,
            "model":      None,
            "latency_ms": 0,
        }

    audio = _wav_bytes_to_float32(wav_bytes)
    t0    = time.perf_counter()

    seg_iter, _ = load_whisper().transcribe(
        audio,
        language=config.WHISPER_LANGUAGE,
        beam_size=config.WHISPER_BEAM_SIZE,
        word_timestamps=True,
        vad_filter=False,
        condition_on_previous_text=False,
    )

    segments: list[dict]   = []
    parts:    list[str]    = []
    futures:  list[Future] = []

    # Each Whisper segment is submitted to Flan-T5 immediately as it is yielded,
    # so correction runs in parallel with Whisper decoding the next segment.
    with ThreadPoolExecutor(max_workers=2) as executor:
        for seg in seg_iter:
            seg_text = seg.text.strip()
            segments.append({
                "start": round(seg.start, 2),
                "end":   round(seg.end, 2),
                "text":  seg_text,
            })
            parts.append(seg.text)
            futures.append(executor.submit(_correct_segment, seg_text))

        # All segments yielded — collect corrected results in original order
        corrected_parts = [fut.result() for fut in futures]

    text           = " ".join(parts).strip()
    corrected_text = " ".join(corrected_parts).strip()
    latency_ms     = round((time.perf_counter() - t0) * 1000)

    log.info("Whisper+Flan-T5 streaming — %d seg(s) | %dms",
             len(segments), latency_ms)

    whisper_result = {
        "text":       text,
        "word_count": len(text.split()) if text else 0,
        "segments":   segments,
    }
    correction_result = {
        "corrected":  corrected_text,
        "enabled":    True,
        "model":      config.FLAN_MODEL,
        "latency_ms": latency_ms,
    }
    return whisper_result, correction_result


# ── Flan-T5 standalone batched correction (used by the WebSocket path) ────────

def correct_text(text: str) -> dict:
    """
    Run Flan-T5 correction on a full transcription text.
    All sentences are batched into a single tokenizer + generate() call.

    Returns:
        {corrected, enabled, model, latency_ms}
    """
    if not config.FLAN_ENABLED:
        return {"corrected": text, "enabled": False,
                "model": None, "latency_ms": 0}
    if not text or not text.strip():
        return {"corrected": text, "enabled": True,
                "model": config.FLAN_MODEL, "latency_ms": 0}

    model, tokenizer = load_flan()
    if model is None or tokenizer is None:
        return {"corrected": text, "enabled": False,
                "model": None, "latency_ms": 0}

    import torch
    device    = next(model.parameters()).device
    sentences = [s.strip() for s in text.split(".") if s.strip()]
    if not sentences:
        return {"corrected": text, "enabled": True,
                "model": config.FLAN_MODEL, "latency_ms": 0}

    prompts = [
        f"Rewrite this with correct grammar and spelling: {s}"
        for s in sentences
    ]

    t_start = time.perf_counter()

    # Single batched tokenize + generate call for all sentences
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
    ).to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=config.FLAN_MAX_TOKENS,
            num_beams=config.FLAN_NUM_BEAMS,
            early_stopping=True,
            no_repeat_ngram_size=3,
        )

    corrected_parts = [
        tokenizer.decode(out, skip_special_tokens=True).rstrip(".").strip()
        for out in outputs
    ]

    latency_ms     = round((time.perf_counter() - t_start) * 1000)
    corrected_text = ". ".join(corrected_parts) + "."

    log.info("Flan-T5 correction done — %d sentences | %dms",
             len(sentences), latency_ms)

    return {
        "corrected":  corrected_text,
        "enabled":    True,
        "model":      config.FLAN_MODEL,
        "latency_ms": latency_ms,
    }