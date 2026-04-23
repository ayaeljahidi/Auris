"""Vosper — Audio utility functions (PyAV-based, no system FFmpeg needed)"""
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


# ── PyAV extraction (replaces FFmpeg subprocess) ───────────────────────────────

def extract_audio(video_bytes: bytes, target_sr: int = 16000) -> bytes:
    """
    Extract 16-kHz mono PCM WAV from any video/audio container via PyAV.
    No system FFmpeg required — PyAV bundles its own codecs.
    Returns WAV bytes. Raises RuntimeError on failure.
    """
    try:
        input_container = av.open(io.BytesIO(video_bytes), mode="r")
    except Exception as exc:
        raise RuntimeError(f"PyAV cannot open input: {exc}")

    # Find the first audio stream
    audio_stream = None
    for stream in input_container.streams:
        if stream.type == "audio":
            audio_stream = stream
            break

    if audio_stream is None:
        raise RuntimeError("No audio stream found in input")

    # Resampler: convert to 16 kHz mono s16
    resampler = av.audio.resampler.AudioResampler(
        format="s16",
        layout="mono",
        rate=target_sr,
    )

    pcm_chunks = []

    try:
        for packet in input_container.demux(audio_stream):
            for frame in packet.decode():
                # Resample frame to target format
                resampled_frames = resampler.resample(frame)
                for resampled in resampled_frames:
                    # to_ndarray() returns int16 for s16 format
                    pcm_chunks.append(resampled.to_ndarray().tobytes())

        # Flush resampler
        flush_frames = resampler.resample(None)
        for resampled in flush_frames:
            pcm_chunks.append(resampled.to_ndarray().tobytes())

    except Exception as exc:
        raise RuntimeError(f"PyAV decode/resample failed: {exc}")
    finally:
        input_container.close()

    full_pcm = b"".join(pcm_chunks)
    return pcm_to_wav(full_pcm, target_sr)


def audio_duration_seconds(pcm: bytes, sample_rate: int) -> float:
    """Duration of raw 16-bit mono PCM in seconds."""
    return len(pcm) / (sample_rate * 2)