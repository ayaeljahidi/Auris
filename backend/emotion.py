"""Auris — Speech Emotion Recognition using Wav2Vec2

Enhancements over v18:
  • Segment-level pooling  — audio is split into overlapping 3-second windows;
    per-segment softmax probabilities are averaged before picking the winner.
    This prevents a single loud/noisy moment from dominating the prediction.
  • calm → neutral collapse — the two classes overlap heavily in Wav2Vec2's
    embedding space.  Calm probability is folded into neutral before the final
    decision, reducing the 8-class confusion to 7 meaningful categories that
    align with the text-emotion model.
  • Reliability threshold raised to 0.55 (was 0.50) — reduces false-confident
    predictions on ambiguous audio.

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
_SEG_LEN_S   = 3      # window length in seconds
_SEG_HOP_S   = 2.0    # hop between windows (was 1.5s — fewer segs, same coverage)
_MIN_SEG_S   = 0.5    # windows shorter than this are skipped
_MAX_SEGS    = 20     # cap: beyond 20 segs accuracy barely improves but cost soars

# Raised reliability threshold (was 0.50)
_RELIABLE_THRESHOLD = 0.55

def _get_persistent_model():
    """Delegate to models.py — single source of truth, no duplicate load."""
    from .models import get_emotion_session
    model, processor, _ = get_emotion_session()
    return model, processor


def _infer_batch(model, processor, segments: list[np.ndarray], sr: int) -> list[np.ndarray]:
    """Run all segments in ONE batched forward pass → list of (8,) numpy arrays."""
    try:
        inputs = processor(
            segments,
            sampling_rate=sr,
            return_tensors="pt",
            padding=True,
        )
        with torch.no_grad():
            logits = model(**inputs).logits          # (N, 8)
            probs  = torch.nn.functional.softmax(logits, dim=1)  # (N, 8)
        return [probs[i].numpy() for i in range(probs.shape[0])]
    except Exception as exc:
        log.warning("Batch inference failed, falling back to single: %s", exc)
        results = []
        for seg in segments:
            try:
                inp = processor(seg, sampling_rate=sr, return_tensors="pt", padding=True)
                with torch.no_grad():
                    lg = model(**inp).logits
                    p  = torch.nn.functional.softmax(lg, dim=1).squeeze()
                if p.dim() == 0:
                    p = p.unsqueeze(0)
                results.append(p.numpy())
            except Exception as e2:
                log.warning("Single segment inference failed: %s", e2)
        return results


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
    Detect emotion from audio using segment-level probability pooling.

    Steps:
      1. Normalise + resample to 16 kHz
      2. Split into overlapping 3-second windows (1.5 s hop)
      3. Run Wav2Vec2 on each window → 8-class softmax
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

    # No hard cap — segment pooling handles any duration efficiently

    # ── Build segments ─────────────────────────────────────────────────────
    seg_len = int(_SEG_LEN_S * sr)
    hop_len = int(_SEG_HOP_S * sr)
    min_len = int(_MIN_SEG_S * sr)
    n_audio = len(audio)

    segments = []
    if n_audio <= seg_len:
        segments.append(audio)
    else:
        start = 0
        while start < n_audio:
            seg = audio[start: start + seg_len]
            if len(seg) >= min_len:
                segments.append(seg)
            start += hop_len

    # Cap to _MAX_SEGS — evenly sample across the audio so we keep coverage
    if len(segments) > _MAX_SEGS:
        indices  = np.linspace(0, len(segments) - 1, _MAX_SEGS, dtype=int)
        segments = [segments[i] for i in indices]

    log.debug("Running %d segments over %.1fs audio (batched)", len(segments), duration_sec)

    # ── Batched inference ─────────────────────────────────────────────────
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
        "Emotion: %s (%.1f%%) [%s] | %d segs | %dms | %.1fx realtime",
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