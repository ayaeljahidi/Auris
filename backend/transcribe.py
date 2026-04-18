"""Vosper — Transcription engines (Vosk + faster-whisper)"""
import json
import logging
import os
import tempfile

import vosk

from .audio import read_wav
from .models import load_vosk, load_whisper
from . import config

log = logging.getLogger("vosper.transcribe")


# ── Vosk ───────────────────────────────────────────────────────────────────────

def transcribe_vosk(wav_bytes: bytes, sample_rate: int = 16000) -> dict:
    """
    Decode WAV with Vosk.

    Returns:
        {text, word_count, words: [{word, start, conf}]}
    """
    pcm, _ = read_wav(wav_bytes)
    rec     = vosk.KaldiRecognizer(load_vosk(), sample_rate)
    rec.SetWords(True)

    chunk_size = 16_000  # 0.5 s at 16 kHz
    results: list[dict] = []

    for offset in range(0, len(pcm), chunk_size):
        chunk = pcm[offset : offset + chunk_size]
        if not chunk:
            break
        if rec.AcceptWaveform(chunk):
            partial = json.loads(rec.Result())
            if partial.get("text"):
                results.append(partial)

    final = json.loads(rec.FinalResult())
    if final.get("text"):
        results.append(final)

    text  = " ".join(r["text"] for r in results).strip()
    words = [
        {
            "word":  w["word"],
            "start": round(w["start"], 3),
            "conf":  round(w.get("conf", 1.0), 2),
        }
        for r in results
        for w in r.get("result", [])
    ]
    return {
        "text":       text,
        "word_count": len(text.split()) if text else 0,
        "words":      words,
    }


# ── faster-whisper ─────────────────────────────────────────────────────────────

def transcribe_whisper(wav_bytes: bytes) -> dict:
    """
    Decode WAV with faster-whisper.

    Returns:
        {text, word_count, segments: [{start, end, text}]}
    """
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(wav_bytes)
        tmp_path = f.name

    try:
        seg_iter, _ = load_whisper().transcribe(
            tmp_path,
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

    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    return {
        "text":       text,
        "word_count": len(text.split()) if text else 0,
        "segments":   segments,
    }
