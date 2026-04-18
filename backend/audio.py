"""Vosper — Audio utility functions"""
import io
import logging
import os
import subprocess
import tempfile
import wave

import numpy as np

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


# ── FFmpeg extraction ──────────────────────────────────────────────────────────

def extract_audio(video_bytes: bytes) -> bytes:
    """
    Extract a 16-kHz mono PCM WAV from any video/audio container via FFmpeg.
    Returns the WAV bytes.  Raises RuntimeError on failure.
    """
    with tempfile.NamedTemporaryFile(suffix=".input", delete=False) as fin:
        fin.write(video_bytes)
        in_path = fin.name

    out_path = in_path + ".wav"
    try:
        result = subprocess.run(
            [
                "ffmpeg", "-y", "-i", in_path,
                "-vn",
                "-acodec", "pcm_s16le",
                "-ar",     "16000",
                "-ac",     "1",
                "-threads", str(config.FFMPEG_THREADS),
                out_path,
            ],
            capture_output=True,
            timeout=config.FFMPEG_TIMEOUT,
        )
        if result.returncode != 0:
            stderr = result.stderr.decode("utf-8", errors="replace")[-600:]
            raise RuntimeError(f"FFmpeg failed: {stderr}")

        with open(out_path, "rb") as f:
            return f.read()

    finally:
        for path in (in_path, out_path):
            try:
                os.unlink(path)
            except OSError:
                pass


def audio_duration_seconds(pcm: bytes, sample_rate: int) -> float:
    """Duration of raw 16-bit mono PCM in seconds."""
    return len(pcm) / (sample_rate * 2)
