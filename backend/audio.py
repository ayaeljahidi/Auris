"""Vosper — Audio utility functions (zero-copy, PyAV-optimized)

P2 Optimisation: Pre-allocate a single numpy output buffer sized for
AUDIO_PREALLOCATE_SAMPLES (default 300 s × 16 kHz = 4.8 M samples ≈ 18 MB).
Frames are written into the buffer with a running pointer; only ONE final
slice/copy is made at the end instead of the previous per-frame list + single
concatenate.  For short audio the pre-allocation is near-zero cost; for long
audio it avoids repeated list-append overhead and the GC churn from keeping
all intermediate ndarrays alive until concat.
"""
import io
import logging
import wave

import numpy as np
import av

from . import config

log = logging.getLogger("vosper.audio")


# ── WAV helpers ────────────────────────────────────────────────────────────────

def read_wav(data: bytes) -> tuple[bytes, int]:
    """Return (raw_pcm, sample_rate) from WAV bytes."""
    with wave.open(io.BytesIO(data), "rb") as w:
        return w.readframes(w.getnframes()), w.getframerate()


def pcm_to_wav(pcm: bytes, sample_rate: int = 16000) -> bytes:
    """Wrap raw PCM bytes in a WAV container."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm)
    return buf.getvalue()


def pcm_to_float32(pcm: bytes) -> np.ndarray:
    """Zero-copy conversion: int16 PCM bytes → float32 numpy array."""
    arr = np.frombuffer(pcm, dtype=np.int16)
    return arr.astype(np.float32, copy=False) * (1.0 / 32768.0)


# ── PyAV extraction (pre-allocated buffer, returns numpy array directly) ───────

def extract_audio_to_numpy(video_bytes: bytes, target_sr: int = 16000) -> np.ndarray:
    """
    Extract 16-kHz mono float32 numpy array from any video/audio container.

    P2: Uses a pre-allocated int16 buffer (AUDIO_PREALLOCATE_SAMPLES long).
    Frames are written in-place; the buffer is sliced once at the end.
    Falls back to dynamic list+concatenate if the audio exceeds the budget.
    No temp files.  Returns array ready for Whisper.
    """
    try:
        input_container = av.open(io.BytesIO(video_bytes), mode="r")
    except Exception as exc:
        raise RuntimeError(f"PyAV cannot open input: {exc}")

    audio_stream = None
    for stream in input_container.streams:
        if stream.type == "audio":
            audio_stream = stream
            break
    if audio_stream is None:
        raise RuntimeError("No audio stream found in input")

    resampler = av.audio.resampler.AudioResampler(
        format="s16", layout="mono", rate=target_sr,
    )

    # ── Pre-allocated path ─────────────────────────────────────────────────
    preallocate = config.AUDIO_PREALLOCATE_SAMPLES
    if preallocate > 0:
        buf      = np.empty(preallocate, dtype=np.int16)
        write_at = 0
        overflow: list[np.ndarray] = []   # only used if audio exceeds budget

        try:
            for packet in input_container.demux(audio_stream):
                for frame in packet.decode():
                    resampled = resampler.resample(frame)
                    for rframe in resampled:
                        nd = rframe.to_ndarray()
                        if nd.ndim > 1:
                            nd = nd.reshape(-1)
                        n = nd.size
                        end = write_at + n
                        if end <= preallocate:
                            buf[write_at:end] = nd
                            write_at = end
                        else:
                            # Spilled — switch to overflow list
                            overflow.append(nd)

            # Flush resampler
            for rframe in resampler.resample(None):
                nd = rframe.to_ndarray()
                if nd.ndim > 1:
                    nd = nd.reshape(-1)
                n = nd.size
                end = write_at + n
                if end <= preallocate and not overflow:
                    buf[write_at:end] = nd
                    write_at = end
                else:
                    overflow.append(nd)

        except Exception as exc:
            raise RuntimeError(f"PyAV decode/resample failed: {exc}")
        finally:
            input_container.close()

        if write_at == 0 and not overflow:
            return np.array([], dtype=np.float32)

        if overflow:
            # Rare: audio longer than AUDIO_PREALLOCATE_SAMPLES
            parts = [buf[:write_at]] + overflow
            pcm_int16 = np.concatenate(parts)
        else:
            # Happy path: single slice, no copy needed beyond the view
            pcm_int16 = buf[:write_at]

        return pcm_int16.astype(np.float32, copy=False) * (1.0 / 32768.0)

    # ── Dynamic fallback (AUDIO_PREALLOCATE_SAMPLES=0) ─────────────────────
    frames: list[np.ndarray] = []
    total_samples = 0

    try:
        for packet in input_container.demux(audio_stream):
            for frame in packet.decode():
                resampled = resampler.resample(frame)
                for rframe in resampled:
                    nd = rframe.to_ndarray()
                    if nd.ndim > 1:
                        nd = nd.reshape(-1)
                    frames.append(nd)
                    total_samples += nd.size

        for rframe in resampler.resample(None):
            nd = rframe.to_ndarray()
            if nd.ndim > 1:
                nd = nd.reshape(-1)
            frames.append(nd)
            total_samples += nd.size

    except Exception as exc:
        raise RuntimeError(f"PyAV decode/resample failed: {exc}")
    finally:
        input_container.close()

    if not frames:
        return np.array([], dtype=np.float32)

    pcm_int16 = np.concatenate(frames)
    return pcm_int16.astype(np.float32, copy=False) * (1.0 / 32768.0)


def extract_audio(video_bytes: bytes, target_sr: int = 16000) -> bytes:
    """
    Legacy wrapper: returns WAV bytes for APIs that need it.
    Prefer extract_audio_to_numpy() for Whisper pipeline.
    """
    arr = extract_audio_to_numpy(video_bytes, target_sr)
    pcm = (arr * 32768.0).astype(np.int16).tobytes()
    return pcm_to_wav(pcm, target_sr)


def audio_duration_seconds(pcm: bytes, sample_rate: int) -> float:
    """Duration of raw 16-bit mono PCM in seconds."""
    return len(pcm) / (sample_rate * 2)


def audio_duration_from_array(arr: np.ndarray, sample_rate: int = 16000) -> float:
    """Duration of float32 numpy array in seconds."""
    return arr.size / sample_rate