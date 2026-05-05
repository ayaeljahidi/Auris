"""Vosper — Audio utility functions (zero-copy, PyAV-optimized)"""
import io
import logging
import wave

import numpy as np
import av

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
    # view as int16 then in-place divide — single allocation
    arr = np.frombuffer(pcm, dtype=np.int16)
    return arr.astype(np.float32, copy=False) * (1.0 / 32768.0)


# ── PyAV extraction (zero-copy, returns numpy array directly) ──────────────────

def extract_audio_to_numpy(video_bytes: bytes, target_sr: int = 16000) -> np.ndarray:
    """
    Extract 16-kHz mono float32 numpy array from any video/audio container.
    Zero-copy path where possible. No temp files. Returns array ready for Whisper.
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

    # Collect frames in a list then concatenate once — avoids per-frame tobytes()
    frames: list[np.ndarray] = []
    total_samples = 0

    try:
        for packet in input_container.demux(audio_stream):
            for frame in packet.decode():
                resampled = resampler.resample(frame)
                for rframe in resampled:
                    nd = rframe.to_ndarray()  # shape: (channels, samples) or (samples,)
                    # Flatten if needed — mono so should be 1D already
                    if nd.ndim > 1:
                        nd = nd.reshape(-1)
                    frames.append(nd)
                    total_samples += nd.size

        # Flush
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

    # Single concatenation + in-place normalization
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