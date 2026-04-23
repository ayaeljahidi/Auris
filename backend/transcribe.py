"""Auris — Transcription engine (faster-whisper + Flan-T5 with batched critique)"""
import io
import logging
import time
import wave

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


# ── Critique logic ────────────────────────────────────────────────────────────

def _should_correct_segment(seg) -> bool:
    """
    Decide whether a Whisper segment needs Flan-T5 correction.
    Uses Whisper's own confidence metrics as a "critique":
      - no_speech_prob:  probability the segment is silence/noise
      - avg_logprob:     average log-probability of tokens (higher = more confident)
      - compression_ratio:  how compressed the text is vs audio (high = hallucination)

    Returns True if the segment looks low-quality and needs correction.
    """
    no_speech_prob = getattr(seg, "no_speech_prob", 0.0)
    avg_logprob = getattr(seg, "avg_logprob", 0.0)
    compression_ratio = getattr(seg, "compression_ratio", 1.0)

    if no_speech_prob > config.CRITIQUE_NO_SPEECH_THRESHOLD:
        return False
    if avg_logprob < config.CRITIQUE_AVG_LOGPROB_THRESHOLD:
        return True
    if compression_ratio > config.CRITIQUE_COMPRESSION_RATIO_MAX:
        return True
    return False


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


# ── Batched Flan-T5 correction for ALL low-confidence segments ────────────────

def _batch_correct_segments(texts: list[str]) -> list[str]:
    """
    Run Flan-T5 correction on multiple segment texts in a SINGLE batched
    generate() call. Each text is split into sentences; all sentences across
    all segments are batched together into one tokenizer + generate() call.
    """
    import torch

    if not texts:
        return []

    model, tokenizer = load_flan()
    if model is None or tokenizer is None:
        return texts

    device = next(model.parameters()).device

    # Build flat prompt list and a parallel segment/sentence index map
    all_prompts: list[str] = []
    mapping: list[tuple[int, int]] = []   # (segment_idx, sentence_idx)
    sentence_counts: list[int] = []       # how many sentences each segment has

    for seg_idx, text in enumerate(texts):
        sentences = [s.strip() for s in text.split(".") if s.strip()]
        if not sentences:
            sentences = [text.strip()]    # keep original if no period found
        sentence_counts.append(len(sentences))
        for sent_idx, sentence in enumerate(sentences):
            all_prompts.append(
                f"Rewrite this with correct grammar and spelling: {sentence}"
            )
            mapping.append((seg_idx, sent_idx))

    if not all_prompts:
        return texts

    # Single batched tokenize + generate for ALL sentences
    inputs = tokenizer(
        all_prompts,
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

    corrected_sentences = [
        tokenizer.decode(out, skip_special_tokens=True).rstrip(".").strip()
        for out in outputs
    ]

    # Reassemble: pre-allocate lists to the exact sentence count per segment
    segment_sentences: list[list[str]] = [
        [""] * cnt for cnt in sentence_counts
    ]
    for (seg_idx, sent_idx), corrected in zip(mapping, corrected_sentences):
        segment_sentences[seg_idx][sent_idx] = corrected

    return [
        ". ".join(sents) + "." if sents else texts[i]
        for i, sents in enumerate(segment_sentences)
    ]


# ── Whisper + Flan-T5 with batched segment correction ─────────────────────────

def transcribe_whisper_with_correction(wav_bytes: bytes) -> tuple[dict, dict]:
    """
    Decode WAV with faster-whisper (in-memory), collect all segments,
    then run Flan-T5 correction in ONE batched call for all low-confidence
    segments. No ThreadPoolExecutor — sequential is faster when both tasks
    are CPU-bound on the same cores.

    Critique logic gates whether Flan-T5 actually runs per segment.
    High-quality segments are "kept" without correction to save CPU.

    Returns:
        (whisper_result, correction_result)
    """
    if not config.FLAN_ENABLED:
        # Run Whisper once and wrap result — don't call transcribe_whisper()
        # separately which would decode the WAV a second time.
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
            segments.append({"start": round(seg.start, 2),
                              "end":   round(seg.end,   2),
                              "text":  seg.text.strip()})
            parts.append(seg.text)
        text = " ".join(parts).strip()
        wr = {"text": text, "word_count": len(text.split()) if text else 0,
              "segments": segments}
        return wr, {
            "corrected":  wr["text"],
            "enabled":    False,
            "model":      None,
            "latency_ms": 0,
            "critique_stats": {"corrected": 0, "kept": 0, "total": 0},
        }

    audio = _wav_bytes_to_float32(wav_bytes)
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
    seg_objs: list = []
    needs_correction: list[bool] = []

    for seg in seg_iter:
        seg_text = seg.text.strip()
        segments.append({
            "start": round(seg.start, 2),
            "end":   round(seg.end, 2),
            "text":  seg_text,
        })
        parts.append(seg.text)
        seg_objs.append(seg)
        needs_correction.append(_should_correct_segment(seg))

    text = " ".join(parts).strip()

    # Collect only segments that failed critique
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

    log.info("Whisper+Flan-T5 batched — %d seg(s) | %d corrected | %d kept | %dms",
             len(segments), corrected_count, kept_count, latency_ms)

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
        "critique_stats": {
            "corrected": corrected_count,
            "kept": kept_count,
            "total": len(segments),
        },
    }
    return whisper_result, correction_result


# ── Flan-T5 standalone batched correction (used by external callers) ──────────

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
    device = next(model.parameters()).device
    sentences = [s.strip() for s in text.split(".") if s.strip()]
    if not sentences:
        return {"corrected": text, "enabled": True,
                "model": config.FLAN_MODEL, "latency_ms": 0}

    prompts = [
        f"Rewrite this with correct grammar and spelling: {s}"
        for s in sentences
    ]

    t_start = time.perf_counter()

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

    latency_ms = round((time.perf_counter() - t_start) * 1000)
    corrected_text = ". ".join(corrected_parts) + "."

    log.info("Flan-T5 correction done — %d sentences | %dms",
             len(sentences), latency_ms)

    return {
        "corrected":  corrected_text,
        "enabled":    True,
        "model":      config.FLAN_MODEL,
        "latency_ms": latency_ms,
    }