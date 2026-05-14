"""Auris — Speech Emotion Recognition using Wav2Vec2

Changes:
  • Window size increased to 5 seconds (was 3s) — better context per segment
  • No hop — segments are purely sequential (no overlap)
    [0s-5s], [5s-10s], [10s-15s], ...
  • No _MAX_SEGS cap — ALL segments are processed
  • Batched inference kept for efficiency
  • calm → neutral collapse kept
  • Reliability threshold kept at 0.55

Label set after collapsing:
  angry · disgust · fear · happy · neutral (+ calm) · sad · surprised
"""
import logging
import time
import torch
import librosa
import numpy as np

from . import config

log = logging.getLogger("auris.emotion")

MODEL_NAME = config.EMOTION_MODEL

# Original 8 labels from the Wav2Vec2 model
_RAW_LABELS  = ["angry", "calm", "disgust", "fear", "happy", "neutral", "sad", "surprised"]
_CALM_IDX    = _RAW_LABELS.index("calm")
_NEUTRAL_IDX = _RAW_LABELS.index("neutral")

# Public label set after calm→neutral collapse (7 emotions, matches text model)
EMOTION_LABELS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprised"]

# Segment settings
# Window increased to 5s — better emotion context, fewer segments
# Hop = window size → pure sequential, zero overlap
# No _MAX_SEGS cap → process ALL segments
_SEG_LEN_S = 5      # window length in seconds (increased from 3s)
_SEG_HOP_S = 5.0    # hop = window size → no overlap, pure sequential
_MIN_SEG_S = 0.5    # windows shorter than this are skipped (tail handling)

# Reliability threshold
_RELIABLE_THRESHOLD = 0.55

# Batch size for inference — process segments in groups to avoid memory issues
_INFER_BATCH_SIZE = 16


def _get_persistent_model():
    """Delegate to models.py — single source of truth, no duplicate load."""
    from .models import get_emotion_session
    model, processor, _ = get_emotion_session()
    return model, processor


def _infer_batch(model, processor, segments: list[np.ndarray], sr: int) -> list[np.ndarray]:
    """
    Run all segments in batched forward passes → list of (8,) numpy arrays.
    Processes _INFER_BATCH_SIZE segments at a time to avoid memory issues
    on long audio (e.g. 10 min = ~120 segments).
    """
    all_results = []

    # Process in sub-batches of _INFER_BATCH_SIZE
    for batch_start in range(0, len(segments), _INFER_BATCH_SIZE):
        batch = segments[batch_start: batch_start + _INFER_BATCH_SIZE]
        try:
            inputs = processor(
                batch,
                sampling_rate=sr,
                return_tensors="pt",
                padding=True,
            )
            with torch.no_grad():
                logits = model(**inputs).logits          # (N, 8)
                probs  = torch.nn.functional.softmax(logits, dim=1)  # (N, 8)
            all_results.extend([probs[i].numpy() for i in range(probs.shape[0])])

        except Exception as exc:
            log.warning("Batch inference failed at batch %d, falling back to single: %s",
                        batch_start, exc)
            # Fallback: process one by one
            for seg in batch:
                try:
                    inp = processor(seg, sampling_rate=sr, return_tensors="pt", padding=True)
                    with torch.no_grad():
                        lg = model(**inp).logits
                        p  = torch.nn.functional.softmax(lg, dim=1).squeeze()
                    if p.dim() == 0:
                        p = p.unsqueeze(0)
                    all_results.append(p.numpy())
                except Exception as e2:
                    log.warning("Single segment inference failed: %s", e2)

    return all_results


def _collapse_calm(probs8: np.ndarray) -> dict:
    """
    Fold calm probability into neutral.
    Returns a 7-entry {label: prob} dict.
    """
    p = probs8.copy()
    p[_NEUTRAL_IDX] += p[_CALM_IDX]
    p[_CALM_IDX] = 0.0
    total = p.sum()
    if total > 0:
        p /= total
    return {lbl: round(float(p[i]), 4)
            for i, lbl in enumerate(_RAW_LABELS)
            if lbl != "calm"}


def detect_emotion_global(audio: np.ndarray, sr: int = 16000) -> dict:
    """
    Detect emotion from audio using full sequential segment processing.

    Steps:
      1. Normalise + resample to 16 kHz
      2. Split into pure sequential 5-second windows (no overlap, no skip)
         [0s-5s], [5s-10s], [10s-15s], ...
      3. Run Wav2Vec2 on ALL windows in batches of _INFER_BATCH_SIZE
      4. Average probabilities across all windows
      5. Collapse calm → neutral → 7-class output
      6. Pick winner; flag reliable when confidence >= 0.55
    """
    t_start = time.perf_counter()

    not_available = {
        "emotion":        "unknown",
        "confidence":     0.0,
        "latency_ms":     0,
        "chunk_count":    0,
        "is_reliable":    False,
        "all_probs":      {},
        "enabled":        config.EMOTION_ENABLED,
        "model":          None,
        "realtime_factor": 0,
        "inference_ms":   0,
    }

    model, processor = _get_persistent_model()
    if model is None or not config.EMOTION_ENABLED:
        return not_available

    # ── Pre-process ────────────────────────────────────────────────────────
    if audio.dtype != np.float32:
        audio = audio.astype(np.float32)
    if audio.size > 0 and np.abs(audio).max() > 1.0:
        audio = audio / 32768.0

    duration_sec = len(audio) / sr if sr > 0 else 0

    if sr != 16000:
        log.debug("Resampling %dHz → 16000Hz", sr)
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        sr = 16000

    # ── Build sequential segments (no overlap, no cap) ─────────────────────
    seg_len = int(_SEG_LEN_S * sr)   # 5s × 16000 = 80000 samples
    hop_len = int(_SEG_HOP_S * sr)   # same as seg_len → no overlap
    min_len = int(_MIN_SEG_S * sr)   # 0.5s minimum to skip tail noise
    n_audio = len(audio)

    segments = []
    if n_audio <= seg_len:
        # Audio shorter than one window → use as-is
        segments.append(audio)
    else:
        start = 0
        while start < n_audio:
            seg = audio[start: start + seg_len]
            if len(seg) >= min_len:
                segments.append(seg)
            start += hop_len   # hop = seg_len → pure sequential

    log.debug(
        "Processing ALL %d segments (%.1fs audio, %ds windows, no overlap)",
        len(segments), duration_sec, _SEG_LEN_S,
    )

    # ── Batched inference (ALL segments, _INFER_BATCH_SIZE at a time) ──────
    t_infer   = time.perf_counter()
    seg_probs = _infer_batch(model, processor, segments, sr)
    infer_ms  = round((time.perf_counter() - t_infer) * 1000)

    if not seg_probs:
        log.error("All segment inferences failed")
        return not_available

    # ── Pool + collapse ────────────────────────────────────────────────────
    avg_probs8 = np.mean(seg_probs, axis=0)   # (8,)
    all_probs  = _collapse_calm(avg_probs8)    # 7-class dict

    best_label = max(all_probs, key=lambda k: all_probs[k])
    confidence = all_probs[best_label]
    is_reliable = confidence >= _RELIABLE_THRESHOLD

    latency_ms  = round((time.perf_counter() - t_start) * 1000)
    speed_ratio = duration_sec / (latency_ms / 1000) if latency_ms > 0 else 0.0

    log.info(
        "Emotion: %s (%.1f%%) [%s] | %d segs (all processed) | %dms | %.1fx realtime",
        best_label.upper(), confidence * 100,
        "reliable" if is_reliable else "low-confidence",
        len(seg_probs), latency_ms, speed_ratio,
    )

    return {
        "emotion":        best_label,
        "confidence":     round(confidence, 4),
        "latency_ms":     latency_ms,
        "chunk_count":    len(seg_probs),
        "is_reliable":    is_reliable,
        "all_probs":      all_probs,
        "enabled":        config.EMOTION_ENABLED,
        "model":          MODEL_NAME,
        "duration_sec":   round(duration_sec, 2),
        "realtime_factor": round(speed_ratio, 2),
        "inference_ms":   infer_ms,
    }


# Backward-compat alias
detect_emotion = detect_emotion_global