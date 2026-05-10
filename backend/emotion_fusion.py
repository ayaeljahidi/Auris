"""Auris — Emotion Fusion Layer

Merges audio (Wav2Vec2) and text (DistilRoBERTa) emotion signals into a single
reliable prediction.

Improvements over v18:
  • calm removed from CANONICAL_LABELS — audio model now collapses calm into
    neutral upstream, so both models share the same 7-class label set.
  • Smarter adaptive weight rules:
      – Agreement bonus: when both models agree on the same label, confidence
        is boosted slightly (up to +0.06) to reward cross-modal consensus.
      – Disagreement penalty: when the models predict different labels AND
        neither is high-confidence, the fused result is flagged low-confidence
        rather than returning a misleadingly certain answer.
      – Word-count gradient: text weight scales linearly with word count
        (5–20 words) instead of a hard threshold flip.
  • Reliability threshold aligned with audio model (0.55 everywhere).
  • Detailed fusion_debug block in every return for easier diagnostics.

Label normalisation (DistilRoBERTa → canonical)
──────────────────────────────────────────────
  anger    → angry      joy      → happy
  disgust  → disgust    neutral  → neutral
  fear     → fear       sadness  → sad
  surprise → surprised
"""
import logging

log = logging.getLogger("auris.emotion_fusion")

# ── Label maps ─────────────────────────────────────────────────────────────────

_TEXT_LABEL_MAP = {
    "anger":    "angry",
    "disgust":  "disgust",
    "fear":     "fear",
    "joy":      "happy",
    "neutral":  "neutral",
    "sadness":  "sad",
    "surprise": "surprised",
    # pass-throughs
    "angry":    "angry",
    "happy":    "happy",
    "sad":      "sad",
    "surprised":"surprised",
}

# 7 canonical labels (calm removed — merged into neutral upstream)
CANONICAL_LABELS = [
    "angry", "disgust", "fear",
    "happy", "neutral", "sad", "surprised",
]

# Reliability threshold (aligned with audio model)
_RELIABLE_THRESHOLD = 0.55

# Agreement bonus and disagreement penalty
_AGREE_BONUS       = 0.06   # added to fused confidence when both models agree
_DISAGREE_LOW_CONF = 0.50   # if neither model reaches this, flag unreliable


# ── Helpers ────────────────────────────────────────────────────────────────────

def _normalise_probs(raw_probs: dict, label_map: dict) -> dict:
    """Map raw labels to canonical; sum when multiple raw labels share one canonical."""
    out: dict[str, float] = {}
    for raw_lbl, prob in raw_probs.items():
        canonical = label_map.get(raw_lbl, raw_lbl)
        out[canonical] = out.get(canonical, 0.0) + prob
    return out


def _word_count_text_weight(word_count: int,
                             low_wc: int = 5, high_wc: int = 20,
                             w_min: float = 0.35, w_max: float = 0.65) -> float:
    """
    Linear ramp: text weight scales from w_min (few words) to w_max (many words).
    Below low_wc → audio only (returns None).
    """
    if word_count < low_wc:
        return None                    # caller handles audio-only fallback
    if word_count >= high_wc:
        return w_max
    frac = (word_count - low_wc) / (high_wc - low_wc)
    return round(w_min + frac * (w_max - w_min), 3)


# ── Public API ─────────────────────────────────────────────────────────────────

def fuse_emotions(
    audio_result: dict,
    text_result: dict,
    audio_weight: float | None = None,
    text_weight:  float | None = None,
) -> dict:
    """
    Merge audio and text emotion predictions with consensus-aware weighting.

    Adaptive weight rules (applied in order when no manual override given):
      1. transcript < 5 words → audio only (text signal too weak)
      2. audio confidence > 0.85 → strong audio signal: 0.60 / 0.40
      3. text confidence > 0.85  → strong text signal:  0.30 / 0.70
      4. 5–20 words: linear ramp from 0.65/0.35 to 0.35/0.65
      5. > 20 words: default 0.35/0.65 (text dominates)

    After weighted merge:
      • Agreement bonus: +0.06 to fused confidence when both predict same label
      • Disagreement penalty: is_reliable = False when neither model is confident
    """
    from . import config

    # ── Check usability ────────────────────────────────────────────────────
    audio_ok = (
        audio_result.get("enabled", False)
        and audio_result.get("emotion", "unknown") != "unknown"
        and audio_result.get("confidence", 0.0) > 0.0
    )
    text_ok = (
        text_result.get("enabled", False)
        and text_result.get("emotion", "unknown") != "unknown"
        and text_result.get("confidence", 0.0) > 0.0
    )

    # ── Single-source fallbacks ────────────────────────────────────────────
    if not audio_ok and not text_ok:
        return _unknown_result(audio_result, text_result)

    if audio_ok and not text_ok:
        log.info("Fusion: audio only (text unavailable)")
        return _wrap(
            emotion=audio_result["emotion"],
            confidence=audio_result["confidence"],
            source="audio_only",
            audio_result=audio_result,
            text_result=text_result,
            weights=(1.0, 0.0),
        )

    if text_ok and not audio_ok:
        canonical = _TEXT_LABEL_MAP.get(text_result["emotion"], text_result["emotion"])
        log.info("Fusion: text only (audio unavailable)")
        return _wrap(
            emotion=canonical,
            confidence=text_result["confidence"],
            source="text_only",
            audio_result=audio_result,
            text_result=text_result,
            weights=(0.0, 1.0),
        )

    # ── Both available: compute weights ───────────────────────────────────
    word_count = text_result.get("word_count", 10)
    audio_conf = audio_result.get("confidence", 0.0)
    text_conf  = text_result.get("confidence", 0.0)

    if audio_weight is None or text_weight is None:
        # Rule 1: very short transcript → audio only
        ramp_w_text = _word_count_text_weight(word_count)
        if ramp_w_text is None:
            log.info("Fusion: short transcript (%d words) → audio only", word_count)
            return _wrap(
                emotion=audio_result["emotion"],
                confidence=audio_result["confidence"],
                source="audio_only_short_text",
                audio_result=audio_result,
                text_result=text_result,
                weights=(1.0, 0.0),
            )
        # Rule 2: strong audio signal
        if audio_conf > 0.85:
            w_audio, w_text = 0.60, 0.40
        # Rule 3: strong text signal
        elif text_conf > 0.85:
            w_audio, w_text = 0.30, 0.70
        # Rules 4/5: word-count ramp
        else:
            w_text  = ramp_w_text
            w_audio = 1.0 - w_text
    else:
        w_audio, w_text = audio_weight, text_weight

    # Normalise
    total   = w_audio + w_text
    w_audio /= total
    w_text  /= total

    # ── Normalise prob dicts to canonical labels ───────────────────────────
    audio_probs = _normalise_probs(audio_result.get("all_probs", {}), {})
    text_probs  = _normalise_probs(text_result.get("all_probs", {}),  _TEXT_LABEL_MAP)

    # ── Weighted merge ─────────────────────────────────────────────────────
    merged: dict[str, float] = {}
    all_labels = set(audio_probs) | set(text_probs) | set(CANONICAL_LABELS)
    for lbl in all_labels:
        merged[lbl] = round(w_audio * audio_probs.get(lbl, 0.0)
                           + w_text  * text_probs.get(lbl,  0.0), 4)

    best_label = max(merged, key=lambda k: merged[k])
    best_conf  = round(merged[best_label], 4)

    # ── Consensus adjustment ───────────────────────────────────────────────
    audio_canonical = audio_result["emotion"]   # already canonical (calm removed)
    text_canonical  = _TEXT_LABEL_MAP.get(text_result["emotion"], text_result["emotion"])
    models_agree    = (audio_canonical == text_canonical)

    if models_agree and audio_canonical == best_label:
        best_conf = round(min(1.0, best_conf + _AGREE_BONUS), 4)
        log.debug("Agreement bonus applied → conf=%.3f", best_conf)

    # Disagreement penalty: if models disagree AND neither is confident,
    # flag the result as unreliable regardless of the merged score.
    neither_confident = (audio_conf < _DISAGREE_LOW_CONF
                         and text_conf < _DISAGREE_LOW_CONF)
    is_reliable = (best_conf >= _RELIABLE_THRESHOLD
                   and not (not models_agree and neither_confident))

    log.info(
        "Fusion: %s (%.1f%%) [%s | agree=%s] "
        "audio=%s(%.1f%%) text=%s(%.1f%%) w=%.0f/%.0f",
        best_label.upper(), best_conf * 100,
        "reliable" if is_reliable else "low-confidence",
        models_agree,
        audio_canonical.upper(), audio_conf * 100,
        text_canonical.upper(),  text_conf  * 100,
        w_audio * 100, w_text * 100,
    )

    return _wrap(
        emotion=best_label,
        confidence=best_conf,
        source="fusion",
        audio_result=audio_result,
        text_result=text_result,
        weights=(round(w_audio, 2), round(w_text, 2)),
        all_probs=merged,
        is_reliable=is_reliable,
        models_agree=models_agree,
    )


# ── Internal helpers ───────────────────────────────────────────────────────────

def _wrap(
    emotion: str,
    confidence: float,
    source: str,
    audio_result: dict,
    text_result: dict,
    weights: tuple[float, float],
    all_probs: dict | None = None,
    is_reliable: bool | None = None,
    models_agree: bool | None = None,
) -> dict:
    if is_reliable is None:
        is_reliable = confidence >= _RELIABLE_THRESHOLD

    text_emotion_canonical = _TEXT_LABEL_MAP.get(
        text_result.get("emotion", "unknown"),
        text_result.get("emotion", "unknown"),
    )

    return {
        # ── Core result ────────────────────────────────────────────────────
        "enabled":     True,
        "emotion":     emotion,
        "confidence":  confidence,
        "is_reliable": is_reliable,
        "all_probs":   all_probs or {},
        "latency_ms":  max(
            audio_result.get("latency_ms", 0),
            text_result.get("latency_ms", 0),
        ),
        # ── Provenance ─────────────────────────────────────────────────────
        "source": source,
        "fusion_weights": {
            "audio": weights[0],
            "text":  weights[1],
        },
        "models_agree": models_agree,
        # ── Per-model breakdown ────────────────────────────────────────────
        "audio_emotion": {
            "emotion":     audio_result.get("emotion", "unknown"),
            "confidence":  audio_result.get("confidence", 0.0),
            "is_reliable": audio_result.get("is_reliable", False),
            "all_probs":   audio_result.get("all_probs", {}),
            "latency_ms":  audio_result.get("latency_ms", 0),
            "model":       audio_result.get("model"),
            "chunk_count": audio_result.get("chunk_count", 0),
        },
        "text_emotion": {
            "emotion":     text_emotion_canonical,
            "confidence":  text_result.get("confidence", 0.0),
            "is_reliable": text_result.get("is_reliable", False),
            "all_probs":   text_result.get("all_probs", {}),
            "latency_ms":  text_result.get("latency_ms", 0),
            "model":       text_result.get("model"),
            "word_count":  text_result.get("word_count", 0),
        },
    }


def _unknown_result(audio_result: dict, text_result: dict) -> dict:
    return {
        "enabled":        False,
        "emotion":        "unknown",
        "confidence":     0.0,
        "is_reliable":    False,
        "all_probs":      {},
        "latency_ms":     0,
        "source":         "none",
        "fusion_weights": {"audio": 0.0, "text": 0.0},
        "models_agree":   None,
        "audio_emotion":  {"emotion": "unknown", "confidence": 0.0},
        "text_emotion":   {"emotion": "unknown", "confidence": 0.0},
    }