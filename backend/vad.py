"""Vosper — MarbleNet VAD (batched ONNX)"""
import io
import logging

import librosa
import numpy as np

from . import config
from .audio import read_wav, pcm_to_wav
from .models import load_marblenet

log = logging.getLogger("vosper.vad")

# Fallback when model is unavailable
_PASSTHROUGH = [{"start": 0.0, "end": 9999.0, "confidence": 1.0}]


def run_vad(
    wav_bytes: bytes,
    sample_rate: int = config.VAD_SAMPLE_RATE,
    threshold: float = config.VAD_THRESHOLD,
    min_speech_ms: int = config.VAD_MIN_SPEECH_MS,
    min_silence_ms: int = config.VAD_MIN_SILENCE_MS,
) -> tuple[list[dict], bytes]:
    """
    Run MarbleNet VAD over wav_bytes.

    Returns:
        (segments, speech_wav)
        segments   — list of {"start", "end", "confidence"}
        speech_wav — WAV bytes containing only speech regions
    """
    session = load_marblenet()
    if session is None:
        return _PASSTHROUGH, wav_bytes

    # ── Load audio ─────────────────────────────────────────────────────────────
    try:
        audio, _ = librosa.load(io.BytesIO(wav_bytes), sr=sample_rate, mono=True)
    except Exception as exc:
        log.error("librosa load error: %s — falling back to raw PCM", exc)
        pcm, _ = read_wav(wav_bytes)
        audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0

    if len(audio) == 0:
        return [], wav_bytes

    # ── Build frame batch ──────────────────────────────────────────────────────
    frame_len = int(0.02 * sample_rate)   # 20 ms
    hop_len   = frame_len
    starts    = np.arange(0, len(audio) - frame_len + 1, hop_len)

    if len(starts) == 0:
        return [], wav_bytes

    batch  = np.stack([audio[s : s + frame_len] for s in starts]).astype(np.float32)
    means  = batch.mean(axis=1, keepdims=True)
    stds   = batch.std(axis=1, keepdims=True) + 1e-8
    batch  = (batch - means) / stds

    # ── ONNX inference ─────────────────────────────────────────────────────────
    input_name = session.get_inputs()[0].name
    try:
        outputs      = session.run(None, {input_name: batch})
        speech_probs = outputs[0][:, 1].tolist()
    except Exception:
        # Frame-by-frame fallback when batched shape is rejected
        speech_probs = []
        for frame in batch:
            out = session.run(None, {input_name: frame.reshape(1, -1)})
            speech_probs.append(float(out[0][0][1]))

    # ── Frames → raw segments ──────────────────────────────────────────────────
    segments: list[dict] = []
    current: dict | None = None

    for s, prob in zip(starts, speech_probs):
        t_start = float(s) / sample_rate
        t_end   = float(s + frame_len) / sample_rate

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
    pad      = int(0.1 * sample_rate)   # 100 ms padding around each segment
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
