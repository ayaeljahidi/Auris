"""Auris — Transcription engine (zero-copy, fused ops, batched critique)"""
import io
import logging
import time
import wave

import numpy as np

from .models import load_whisper, load_flan
from . import config

log = logging.getLogger("auris.transcribe")


# ── Zero-copy: WAV bytes → float32 numpy (single pass) ────────────────────────

def _wav_bytes_to_float32(wav_bytes: bytes) -> np.ndarray:
    """Parse WAV header and return float32 array in one pass."""
    with wave.open(io.BytesIO(wav_bytes), "rb") as w:
        pcm = w.readframes(w.getnframes())
    # Single allocation: view as int16, cast to float32, scale in-place
    arr = np.frombuffer(pcm, dtype=np.int16)
    return arr.astype(np.float32, copy=False) * (1.0 / 32768.0)


# ── Critique logic (unchanged semantics, inlined for speed) ───────────────────

def _should_correct_segment(seg) -> bool:
    """Return True if segment quality is low enough to need Flan-T5."""
    if getattr(seg, "no_speech_prob", 0.0) > config.CRITIQUE_NO_SPEECH_THRESHOLD:
        return False
    if getattr(seg, "avg_logprob", 0.0) < config.CRITIQUE_AVG_LOGPROB_THRESHOLD:
        return True
    if getattr(seg, "compression_ratio", 1.0) > config.CRITIQUE_COMPRESSION_RATIO_MAX:
        return True
    return False


# ── faster-whisper (accepts numpy array directly) ─────────────────────────────

def transcribe_whisper(audio: np.ndarray) -> dict:
    """
    Decode with faster-whisper using a pre-loaded float32 array.

    Returns:
        {text, word_count, segments: [{start, end, text}]}
    """
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
        segments.append({"start": round(seg.start, 2), "end": round(seg.end, 2), "text": txt})
        parts.append(seg.text)

    text = " ".join(parts).strip()
    return {
        "text": text,
        "word_count": len(text.split()) if text else 0,
        "segments": segments,
    }


# ── Fast Flan-T5: segment-level correction without sentence splitting ──────────

def _batch_correct_segments(texts: list[str]) -> list[str]:
    """
    Run Flan-T5 on segment texts directly (no sentence splitting).
    Single batched tokenizer + generate() call.
    """
    import torch

    if not texts:
        return []

    model, tokenizer = load_flan()
    if model is None or tokenizer is None:
        return texts

    device = next(model.parameters()).device

    # Direct segment-level prompts — skip expensive sentence splitting
    prompts = [f"Fix grammar: {t}" for t in texts]

    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=256,  # Reduced from 512 — segments are shorter
    ).to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=config.FLAN_MAX_TOKENS,
            num_beams=config.FLAN_NUM_BEAMS,
            early_stopping=True,
            no_repeat_ngram_size=2,  # Reduced from 3 — faster decoding
        )

    return [
        tokenizer.decode(out, skip_special_tokens=True).strip()
        for out in outputs
    ]


# ── Whisper + Flan-T5 with batched segment correction ─────────────────────────

def transcribe_whisper_with_correction(audio: np.ndarray) -> tuple[dict, dict]:
    """
    Decode with Whisper, then run Flan-T5 in ONE batched call for low-confidence
    segments. Accepts numpy array directly — zero copy from extraction.

    Returns:
        (whisper_result, correction_result)
    """
    t0 = time.perf_counter()

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
    needs_correction: list[bool] = []

    for seg in seg_iter:
        txt = seg.text.strip()
        segments.append({"start": round(seg.start, 2), "end": round(seg.end, 2), "text": txt})
        parts.append(seg.text)
        needs_correction.append(_should_correct_segment(seg))

    text = " ".join(parts).strip()

    # Early exit: no correction needed at all
    if not config.FLAN_ENABLED or not any(needs_correction):
        latency_ms = round((time.perf_counter() - t0) * 1000)
        wr = {"text": text, "word_count": len(text.split()) if text else 0, "segments": segments}
        return wr, {
            "corrected": wr["text"],
            "enabled": config.FLAN_ENABLED,
            "model": config.FLAN_MODEL if config.FLAN_ENABLED else None,
            "latency_ms": latency_ms,
            "critique_stats": {"corrected": 0, "kept": len(segments), "total": len(segments)},
        }

    to_correct_texts = [parts[i] for i, need in enumerate(needs_correction) if need]
    to_correct_indices = [i for i, need in enumerate(needs_correction) if need]

    corrected_parts = parts.copy()

    if to_correct_texts:
        batch_results = _batch_correct_segments(to_correct_texts)
        for idx_in_batch, seg_idx in enumerate(to_correct_indices):
            corrected_parts[seg_idx] = batch_results[idx_in_batch]

    corrected_text = " ".join(corrected_parts).strip()
    latency_ms = round((time.perf_counter() - t0) * 1000)

    corrected_count = sum(1 for p, o in zip(corrected_parts, parts) if p != o)
    kept_count = len(parts) - corrected_count

    log.info("Whisper+Flan-T5 — %d seg(s) | %d corrected | %d kept | %dms",
             len(segments), corrected_count, kept_count, latency_ms)

    whisper_result = {
        "text": text,
        "word_count": len(text.split()) if text else 0,
        "segments": segments,
    }
    correction_result = {
        "corrected": corrected_text,
        "enabled": True,
        "model": config.FLAN_MODEL,
        "latency_ms": latency_ms,
        "critique_stats": {
            "corrected": corrected_count,
            "kept": kept_count,
            "total": len(segments),
        },
    }
    return whisper_result, correction_result


# ── Standalone text correction (segment-level, no splitting) ──────────────────

def correct_text(text: str) -> dict:
    """Run Flan-T5 correction on full text as single segment."""
    if not config.FLAN_ENABLED or not text or not text.strip():
        return {"corrected": text, "enabled": config.FLAN_ENABLED,
                "model": config.FLAN_MODEL if config.FLAN_ENABLED else None, "latency_ms": 0}

    model, tokenizer = load_flan()
    if model is None or tokenizer is None:
        return {"corrected": text, "enabled": False, "model": None, "latency_ms": 0}

    import torch
    device = next(model.parameters()).device

    t_start = time.perf_counter()

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

    corrected = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    latency_ms = round((time.perf_counter() - t_start) * 1000)

    log.info("Flan-T5 correction done — %dms", latency_ms)

    return {
        "corrected": corrected,
        "enabled": True,
        "model": config.FLAN_MODEL,
        "latency_ms": latency_ms,
    }