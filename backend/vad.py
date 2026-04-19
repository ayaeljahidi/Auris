"""Vosper — Silero VAD (stateful ONNX)"""
import io
import logging

import numpy as np

from . import config
from .audio import read_wav, pcm_to_wav
from .models import load_marblenet

log = logging.getLogger("vosper.vad")

# Fallback when model is unavailable
_PASSTHROUGH = [{"start": 0.0, "end": 9999.0, "confidence": 1.0}]

# Silero requires exactly 512 samples per chunk at 16 kHz
CHUNK_SIZE = 512


def run_vad(
    wav_bytes: bytes,
    sample_rate: int = config.VAD_SAMPLE_RATE,
    threshold: float = config.VAD_THRESHOLD,
    min_speech_ms: int = config.VAD_MIN_SPEECH_MS,
    min_silence_ms: int = config.VAD_MIN_SILENCE_MS,
) -> tuple[list[dict], bytes]:
    """
    Run Silero VAD over wav_bytes.

    Returns:
        (segments, speech_wav)
        segments   — list of {"start", "end", "confidence"}
        speech_wav — WAV bytes containing only speech regions
    """
    session = load_marblenet()
    if session is None:
        return _PASSTHROUGH, wav_bytes

    # ── Load audio as float32 PCM ──────────────────────────────────────────────
    try:
        pcm_bytes, _ = read_wav(wav_bytes)
        audio = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    except Exception as exc:
        log.error("audio load error: %s — passthrough", exc)
        return _PASSTHROUGH, wav_bytes

    if len(audio) == 0:
        return [], wav_bytes

    # ── Silero stateful inference ──────────────────────────────────────────────
    # Initialise LSTM hidden states
    h  = np.zeros((2, 1, 64), dtype=np.float32)
    c  = np.zeros((2, 1, 64), dtype=np.float32)
    sr = np.array(sample_rate, dtype=np.int64)

    speech_probs: list[float] = []
    starts: list[int] = []

    for start in range(0, len(audio) - CHUNK_SIZE + 1, CHUNK_SIZE):
        chunk = audio[start : start + CHUNK_SIZE].reshape(1, -1)  # (1, 512)

        out, h, c = session.run(
            ["output", "hn", "cn"],
            {"input": chunk, "sr": sr, "h": h, "c": c},
        )

        speech_probs.append(float(out[0][0]))
        starts.append(start)

    if not speech_probs:
        return [], wav_bytes

    # ── Probs → segments ───────────────────────────────────────────────────────
    segments: list[dict] = []
    current: dict | None = None
    chunk_dur = CHUNK_SIZE / sample_rate          # seconds per chunk

    for start_sample, prob in zip(starts, speech_probs):
        t_start = start_sample / sample_rate
        t_end   = t_start + chunk_dur

        if prob > threshold:
            if current is None:
                current = {"start": t_start, "end": t_end, "probs": [prob]}
            else:
                current["end"] = t_end
                current["probs"].append(prob)
        else:
            if current is not None:
                if (current["end"] - current["start"]) * 1000 >= min_speech_ms:
                    segments.append(_finalise(current))
                current = None

    if current is not None and (current["end"] - current["start"]) * 1000 >= min_speech_ms:
        segments.append(_finalise(current))

    # ── Merge close segments ───────────────────────────────────────────────────
    segments = _merge_segments(segments, min_silence_ms)

    # ── Extract speech PCM ─────────────────────────────────────────────────────
    full_pcm = np.frombuffer(read_wav(wav_bytes)[0], dtype=np.int16)
    pad      = int(0.1 * sample_rate)
    parts    = []

    for seg in segments:
        s = max(0,             int(seg["start"] * sample_rate) - pad)
        e = min(len(full_pcm), int(seg["end"]   * sample_rate) + pad)
        if e > s:
            parts.append(full_pcm[s:e])

    speech_pcm = np.concatenate(parts).tobytes() if parts else full_pcm.tobytes()
    log.info("VAD: %d speech segment(s) detected", len(segments))
    return segments, pcm_to_wav(speech_pcm, sample_rate)


# ── Private helpers ────────────────────────────────────────────────────────────

def _finalise(seg: dict) -> dict:
    return {
        "start":      round(seg["start"], 3),
        "end":        round(seg["end"],   3),
        "confidence": round(float(np.mean(seg["probs"])), 3),
    }


def _merge_segments(segments: list[dict], min_silence_ms: int) -> list[dict]:
    if len(segments) <= 1:
        return segments
    merged = [segments[0].copy()]
    for seg in segments[1:]:
        gap_ms = (seg["start"] - merged[-1]["end"]) * 1000
        if gap_ms < min_silence_ms:
            merged[-1]["end"]        = seg["end"]
            merged[-1]["confidence"] = round(
                (merged[-1]["confidence"] + seg["confidence"]) / 2, 3
            )
        else:
            merged.append(seg.copy())
    return merged