"""Auris — Text-based Emotion Detection (DistilRoBERTa)

Uses j-hartmann/emotion-english-distilroberta-base
7 classes: anger, disgust, fear, joy, neutral, sadness, surprise
~300MB, CPU-only, no compilation needed.

Improvements over v18:
  • Reliability threshold raised to 0.55 (was 0.45) — consistent with audio.
  • Minimum word count for reliability raised to 4 (was 3).
  • Long-text chunking: transcripts > 512 tokens are split into 256-token
    overlapping chunks; scores are averaged before returning.  Previously,
    text was silently truncated at 512 chars, discarding the second half.
"""
import logging
import time

import torch
from transformers import pipeline as hf_pipeline

from . import config

log = logging.getLogger("auris.text_emotion")

# Thresholds (aligned with audio model)
_RELIABLE_THRESHOLD = 0.55
_MIN_WORDS_RELIABLE = 4

# Chunking for long transcripts
_CHUNK_WORDS = 80    # ~256 tokens at ~3.2 chars/token
_CHUNK_HOP   = 60    # 25-word overlap between chunks


def load_text_emotion_model():
    """Delegate to models.py — single source of truth, no duplicate load."""
    from .models import load_text_emotion_model as _models_load
    return _models_load()


def _build_chunks(text: str) -> list[str]:
    """
    Split transcript into word-based overlapping chunks so we never silently
    discard content.  If the text fits in _CHUNK_WORDS words, return as-is.
    """
    words = text.split()
    if len(words) <= _CHUNK_WORDS:
        return [text]

    chunks = []
    start = 0
    while start < len(words):
        chunk = " ".join(words[start: start + _CHUNK_WORDS])
        chunks.append(chunk)
        start += _CHUNK_HOP
    return chunks


def _run_pipe(pipe, chunks: list[str]) -> list[dict]:
    """
    Run the HuggingFace pipeline on every chunk and average the per-label
    probabilities.  Returns a list of {label, score} dicts for the final
    averaged distribution.
    """
    import numpy as np

    label_accum: dict[str, list[float]] = {}

    for chunk in chunks:
        raw = pipe(chunk)                              # chunking already keeps each chunk within token limit
        scores = raw[0] if isinstance(raw[0], list) else raw
        for item in scores:
            lbl = item["label"].lower()
            label_accum.setdefault(lbl, []).append(float(item["score"]))

    # Average across chunks
    averaged = [
        {"label": lbl, "score": float(np.mean(vals))}
        for lbl, vals in label_accum.items()
    ]
    return averaged


def detect_emotion_from_text(text: str) -> dict:
    """
    Detect emotion from transcript text using DistilRoBERTa.

    Long transcripts are split into overlapping chunks; scores are averaged.
    Reliability requires confidence >= 0.55 AND word_count >= 4.
    """
    _not_available = {
        "emotion":    "unknown",
        "confidence": 0.0,
        "latency_ms": 0,
        "is_reliable": False,
        "all_probs":  {},
        "enabled":    config.TEXT_EMOTION_ENABLED,
        "model":      None,
    }

    if not config.TEXT_EMOTION_ENABLED:
        return _not_available

    if not text or not text.strip():
        log.debug("Text emotion skipped — empty transcript")
        return {**_not_available, "error": "empty transcript"}

    word_count = len(text.split())

    pipe = load_text_emotion_model()
    if pipe is None:
        return _not_available

    t_start = time.perf_counter()
    try:
        chunks  = _build_chunks(text)
        scores  = _run_pipe(pipe, chunks)

        all_probs = {item["label"]: round(float(item["score"]), 4) for item in scores}

        best        = max(scores, key=lambda x: x["score"])
        emotion_label = best["label"].lower()
        confidence    = round(float(best["score"]), 4)
        is_reliable   = confidence >= _RELIABLE_THRESHOLD and word_count >= _MIN_WORDS_RELIABLE

        latency_ms = round((time.perf_counter() - t_start) * 1000)

        log.info(
            "Text emotion: %s (%.1f%%) [%s] | %dms | %d words | %d chunk(s)",
            emotion_label.upper(), confidence * 100,
            "reliable" if is_reliable else "low-confidence",
            latency_ms, word_count, len(chunks),
        )

        return {
            "emotion":     emotion_label,
            "confidence":  confidence,
            "latency_ms":  latency_ms,
            "is_reliable": is_reliable,
            "all_probs":   all_probs,
            "enabled":     True,
            "model":       config.TEXT_EMOTION_MODEL,
            "word_count":  word_count,
            "chunk_count": len(chunks),
        }

    except Exception as exc:
        latency_ms = round((time.perf_counter() - t_start) * 1000)
        log.error("Text emotion inference failed: %s", exc)
        return {**_not_available, "latency_ms": latency_ms, "error": str(exc)}