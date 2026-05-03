"""Auris — Emotion detection with Silero VAD gating

Speed-up strategy
─────────────────
Old path:  split audio into N fixed chunks → run every chunk through wav2vec2
New path:  run Silero VAD (~2 ms) first → build chunks ONLY around speech regions
           → skip silence entirely → wav2vec2 only sees speech

Typical gain on a 95 s video call (≈ 40 % silence):
    Before: 7 chunks × 4 400 ms  = 30 800 ms
    After:  4 speech chunks × 4 400 ms = 17 600 ms  (–43 %)
    VAD overhead: < 5 ms  (negligible)

VAD model: Silero VAD (snakers4/silero-vad), ~2 MB PyTorch, pure CPU.
Loaded once at module import, kept persistent alongside the ONNX session.

CRITICAL FIX (v2):
  1. REMOVED ThreadPoolExecutor — ONNX InferenceSession is NOT thread-safe.
     Parallel threads contend for the same session context, causing massive
     slowdown. Sequential inference is faster for CPU-bound ONNX.
  2. CHUNK_SECONDS = 10 (was 15). No merging of adjacent VAD regions.
     Each speech region becomes its own chunk capped at 10 s.
     Prevents "fat chunks" (~24 s each) that balloon per-chunk inference time.
"""
import logging
import time
import numpy as np
import torch

from . import config

log = logging.getLogger("auris.emotion")

try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ort = None
    ONNX_AVAILABLE = False
    log.warning("ONNX Runtime not available")

# ── Chunking parameters ───────────────────────────────────────────────────────
CHUNK_SECONDS     = 10.0   # max length of one emotion chunk (seconds)
MIN_CHUNK_SECONDS = 1.0    # ignore speech regions shorter than this
UNIFORM_THRESHOLD = 0.20   # minimum confidence to be "reliable"

# VAD parameters
VAD_THRESHOLD         = 0.35   # Silero speech probability threshold
VAD_MIN_SPEECH_MS     = 250    # ignore speech bursts shorter than this
VAD_MIN_SILENCE_MS    = 200    # gaps shorter than this are merged into speech
VAD_WINDOW_SIZE_SAMPLES = 512  # Silero operates on 512-sample frames at 16 kHz

# ── Persistent singletons (loaded once at module import) ──────────────────────
_emotion_session: "ort.InferenceSession | None" = None
_emotion_labels:  list[str] = []
_session_loaded:  bool = False

_vad_model      = None
_vad_utils      = None
_vad_loaded:    bool = False


# ── VAD loader ────────────────────────────────────────────────────────────────

def _load_vad() -> tuple:
    """Load Silero VAD model once; return (model, get_speech_timestamps) or (None, None)."""
    global _vad_model, _vad_utils, _vad_loaded

    if _vad_loaded:
        return _vad_model, _vad_utils

    try:
        vad_model, vad_utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
            onnx=False,
        )
        vad_model.eval()

        # vad_utils is a 5-tuple; index-0 is get_speech_timestamps
        get_speech_timestamps = vad_utils[0]

        _vad_model = vad_model
        _vad_utils = get_speech_timestamps
        _vad_loaded = True
        log.info("✓ Silero VAD loaded (persistent, CPU)")
        return _vad_model, _vad_utils

    except Exception as exc:
        log.warning("Silero VAD unavailable — VAD gating disabled: %s", exc)
        _vad_loaded = True          # mark as attempted so we don't retry every request
        return None, None


# ── Emotion session loader ────────────────────────────────────────────────────

def _get_persistent_session() -> tuple:
    global _emotion_session, _emotion_labels, _session_loaded

    if _session_loaded:
        return _emotion_session, _emotion_labels

    try:
        from .models import get_emotion_session
        session, labels = get_emotion_session()

        if session is not None:
            _emotion_session = session
            _emotion_labels  = labels if labels else [
                "angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"
            ]
            _session_loaded = True
            log.info("✓ Persistent ONNX emotion session acquired")
            return _emotion_session, _emotion_labels
        else:
            log.error("Failed to get emotion session")
            _session_loaded = True
            return None, None

    except Exception as exc:
        log.error("Error getting emotion session: %s", exc)
        _session_loaded = True
        return None, None


# Load both models at module import
PERSISTENT_SESSION, PERSISTENT_LABELS = _get_persistent_session()
_load_vad()   # pre-warm VAD so first request has zero loading overhead


# ── VAD gating: build speech-only chunks ─────────────────────────────────────

def _get_speech_chunks(audio: np.ndarray, sr: int) -> list[tuple[float, float, np.ndarray]]:
    """
    Use Silero VAD to identify speech regions, then slice the audio into
    chunks of at most CHUNK_SECONDS covering only those regions.

    Each speech region is capped independently at CHUNK_SECONDS.
    NO merging of adjacent regions — prevents "fat chunk" bug.

    Returns a list of (start_sec, end_sec, chunk_array) covering speech only.
    Falls back to fixed chunking if VAD is unavailable.
    """
    vad_model, get_speech_timestamps = _vad_model, _vad_utils

    # ── Fallback: no VAD available ─────────────────────────────────────────
    if vad_model is None or get_speech_timestamps is None:
        log.debug("VAD unavailable — using fixed chunking")
        return _fixed_chunks(audio, sr)

    # ── Run Silero VAD ─────────────────────────────────────────────────────
    try:
        t_vad = time.perf_counter()

        # Silero expects a 1-D float32 tensor
        audio_tensor = torch.from_numpy(audio.astype(np.float32))

        vad_model.reset_states()
        with torch.no_grad():
            speech_timestamps = get_speech_timestamps(
                audio_tensor,
                vad_model,
                sampling_rate=sr,
                threshold=VAD_THRESHOLD,
                min_speech_duration_ms=VAD_MIN_SPEECH_MS,
                min_silence_duration_ms=VAD_MIN_SILENCE_MS,
                window_size_samples=VAD_WINDOW_SIZE_SAMPLES,
                return_seconds=False,   # returns sample indices
            )

        vad_ms = round((time.perf_counter() - t_vad) * 1000)
        total_samples  = len(audio)
        speech_samples = sum(ts["end"] - ts["start"] for ts in speech_timestamps)
        silence_ratio  = 1.0 - (speech_samples / max(total_samples, 1))

        log.info(
            "VAD: %d speech region(s) | %.1f%% silence skipped | %d ms",
            len(speech_timestamps),
            silence_ratio * 100,
            vad_ms,
        )

        if not speech_timestamps:
            log.info("VAD: no speech detected — returning empty result")
            return []

        # ── Build chunks: each region capped at CHUNK_SECONDS ──────────────
        # FIX: No merging across regions. Each VAD region is split independently
        # if it exceeds CHUNK_SECONDS. This prevents "fat chunks" that cause
        # per-chunk inference time to explode (e.g. 24 s → 8 729 ms).
        chunks: list[tuple[float, float, np.ndarray]] = []
        max_chunk_samples = int(CHUNK_SECONDS * sr)

        for ts in speech_timestamps:
            region_start = ts["start"]
            region_end   = ts["end"]
            region_len   = region_end - region_start

            if region_len <= max_chunk_samples:
                # Region fits in one chunk
                chunk = audio[region_start:region_end]
                dur = len(chunk) / sr
                if dur >= MIN_CHUNK_SECONDS:
                    chunks.append((region_start / sr, region_end / sr, chunk))
            else:
                # Region too long — split into CHUNK_SECONDS pieces
                pos = region_start
                while pos < region_end:
                    end = min(pos + max_chunk_samples, region_end)
                    chunk = audio[pos:end]
                    dur = len(chunk) / sr
                    if dur >= MIN_CHUNK_SECONDS:
                        chunks.append((pos / sr, end / sr, chunk))
                    if end == region_end:
                        break
                    pos += max_chunk_samples

        log.info(
            "VAD chunking: %d chunk(s) from %d region(s) | silence skipped: %.1f%%",
            len(chunks), len(speech_timestamps), silence_ratio * 100,
        )
        return chunks

    except Exception as exc:
        log.warning("VAD chunking failed (%s) — falling back to fixed chunks", exc)
        return _fixed_chunks(audio, sr)


def _fixed_chunks(audio: np.ndarray, sr: int) -> list[tuple[float, float, np.ndarray]]:
    """Fixed-size chunking fallback (original behaviour, no VAD)."""
    chunk_samples = int(CHUNK_SECONDS * sr)
    total_samples = len(audio)

    if total_samples <= chunk_samples:
        dur = total_samples / sr
        return [(0.0, dur, audio)] if dur >= MIN_CHUNK_SECONDS else []

    chunks = []
    pos = 0
    while pos < total_samples:
        end   = min(pos + chunk_samples, total_samples)
        chunk = audio[pos:end]
        dur   = len(chunk) / sr
        if dur >= MIN_CHUNK_SECONDS:
            chunks.append((pos / sr, end / sr, chunk))
        if end == total_samples:
            break
        pos += chunk_samples

    return chunks


# ── ONNX inference (SEQUENTIAL — no ThreadPool) ──────────────────────────────

def _process_single_chunk(args: tuple) -> tuple[int, np.ndarray]:
    """Run one chunk through the ONNX session and return (idx, probs)."""
    chunk_audio, chunk_idx = args

    try:
        if chunk_audio.dtype != np.float32:
            chunk_audio = chunk_audio.astype(np.float32)
        if np.abs(chunk_audio).max() > 1.0:
            chunk_audio = chunk_audio / 32768.0
        if chunk_audio.ndim == 1:
            chunk_audio = chunk_audio.reshape(1, -1)

        input_name = PERSISTENT_SESSION.get_inputs()[0].name
        outputs    = PERSISTENT_SESSION.run(None, {input_name: chunk_audio})
        logits     = outputs[0]

        probs = np.exp(logits[0] - logits[0].max())
        probs = probs / probs.sum()
        return chunk_idx, probs

    except Exception as exc:
        log.error("Chunk %d error: %s", chunk_idx, exc)
        n = len(PERSISTENT_LABELS) if PERSISTENT_LABELS else 7
        return chunk_idx, np.ones(n) / n


def _predict_sequential(chunks: list) -> list[np.ndarray]:
    """
    Run all chunks SEQUENTIALLY through the persistent ONNX session.

    CRITICAL: ONNX InferenceSession is NOT thread-safe for concurrent .run().
    ThreadPoolExecutor causes lock contention and context-switch overhead,
    making parallel slower than sequential on CPU. Each chunk is fast
    (~200-400 ms), so sequential is optimal here.
    """
    if not chunks or PERSISTENT_SESSION is None:
        return []

    results: list[np.ndarray] = []
    for idx, (_, _, chunk_audio) in enumerate(chunks):
        _, probs = _process_single_chunk((chunk_audio, idx))
        results.append(probs)
    return results


def _aggregate(chunk_probs: list, chunk_durations: list) -> np.ndarray:
    """Duration-weighted average of per-chunk probability vectors."""
    if not chunk_probs:
        return np.array([])
    weights     = np.array(chunk_durations, dtype=np.float32)
    weights    /= weights.sum()
    probs_matrix = np.vstack(chunk_probs)
    return np.dot(weights, probs_matrix)


# ── Public API ────────────────────────────────────────────────────────────────

def detect_emotion_global(audio: np.ndarray, sr: int = 16000) -> dict:
    """
    Detect emotion on the full audio array using VAD-gated chunking.

    Flow:
        1. Silero VAD  (~2 ms)  → speech timestamps
        2. Build chunks from speech regions only, each capped at 10 s
           (NO merging across regions — prevents fat-chunk bug)
        3. SEQUENTIAL ONNX inference on speech chunks
        4. Duration-weighted aggregation → single emotion label

    Args:
        audio: float32 numpy array, mono, 16 kHz
        sr:    sample rate (must be 16000 for Silero VAD)

    Returns:
        dict with keys: emotion, confidence, latency_ms, chunk_count,
                        is_reliable, all_probs, vad_silence_skipped,
                        realtime_factor, inference_ms, avg_chunk_ms, enabled
    """
    t_start = time.perf_counter()

    _not_available = {
        "emotion":    "unknown",
        "confidence": 0.0,
        "latency_ms": 0,
        "chunk_count": 0,
        "is_reliable": False,
        "all_probs": {},
        "vad_silence_skipped": False,
        "enabled": config.EMOTION_ENABLED,
    }

    if PERSISTENT_SESSION is None or not config.EMOTION_ENABLED:
        return _not_available

    labels = PERSISTENT_LABELS if PERSISTENT_LABELS else [
        "angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"
    ]

    # Normalise
    if audio.dtype != np.float32:
        audio = audio.astype(np.float32)
    if audio.size > 0 and np.abs(audio).max() > 1.0:
        audio = audio / 32768.0

    duration_sec = len(audio) / sr

    # ── Step 1: VAD-gated chunking ─────────────────────────────────────────
    t_vad_start  = time.perf_counter()
    chunks       = _get_speech_chunks(audio, sr)
    vad_ms       = round((time.perf_counter() - t_vad_start) * 1000)
    vad_was_used = (_vad_model is not None and _vad_utils is not None)

    if not chunks:
        log.info("Emotion: no speech chunks after VAD — returning unknown")
        return {**_not_available, "latency_ms": round((time.perf_counter() - t_start) * 1000)}

    chunk_count = len(chunks)

    # ── Step 2: Sequential ONNX inference ─────────────────────────────────
    t_infer     = time.perf_counter()
    chunk_probs = _predict_sequential(chunks)
    infer_ms    = round((time.perf_counter() - t_infer) * 1000)

    if not chunk_probs:
        return {**_not_available, "latency_ms": round((time.perf_counter() - t_start) * 1000)}

    # ── Step 3: Aggregate ─────────────────────────────────────────────────
    chunk_durations = [end - start for start, end, _ in chunks]
    aggregated      = _aggregate(chunk_probs, chunk_durations)
    pred_idx        = int(np.argmax(aggregated))
    emotion         = labels[pred_idx] if pred_idx < len(labels) else "unknown"
    confidence      = float(aggregated[pred_idx])
    latency_ms      = round((time.perf_counter() - t_start) * 1000)
    is_reliable     = confidence >= UNIFORM_THRESHOLD
    all_probs       = {labels[i]: round(float(aggregated[i]), 4) for i in range(len(labels))}

    speed_ratio        = duration_sec / (latency_ms / 1000) if latency_ms > 0 else 0.0
    avg_chunk_ms       = infer_ms / chunk_count if chunk_count > 0 else 0.0
    speech_covered_sec = sum(e - s for s, e, _ in chunks)
    silence_skipped_sec = max(0.0, duration_sec - speech_covered_sec)
    silence_pct        = (silence_skipped_sec / duration_sec * 100) if duration_sec > 0 else 0.0

    # ── Logging ───────────────────────────────────────────────────────────
    print()
    print("╔" + "═" * 67 + "╗")
    print("║" + " ⚡ EMOTION (Persistent ONNX + Silero VAD)".center(67) + "║")
    print("╠" + "═" * 67 + "╣")
    print(f"║  {emotion.upper():<12} {confidence:>6.1%}  {'✓ RELIABLE' if is_reliable else '⚠ LOW CONFIDENCE'}".ljust(67) + "║")
    print(f"║  Duration: {duration_sec:.1f}s  |  Speech chunks: {chunk_count}  |  Total: {latency_ms}ms".ljust(67) + "║")
    print(f"║  VAD: {vad_ms}ms overhead  |  Silence skipped: {silence_skipped_sec:.1f}s ({silence_pct:.0f}%)".ljust(67) + "║")
    print(f"║  Inference: {infer_ms}ms  |  Per chunk: {avg_chunk_ms:.0f}ms  |  {chunk_count} chunks sequential".ljust(67) + "║")
    print(f"║  Speed: {speed_ratio:.1f}x realtime  |  VAD gating: {'ON' if vad_was_used else 'OFF (fallback)'}".ljust(67) + "║")
    print("╚" + "═" * 67 + "╝")

    log.info(
        "✅ Emotion: %s (%.1f%%) | %d chunks | %.1fs | %dms | %.2fx | "
        "VAD %dms | silence skipped %.1fs (%.0f%%)",
        emotion, confidence * 100, chunk_count, duration_sec, latency_ms,
        speed_ratio, vad_ms, silence_skipped_sec, silence_pct,
    )

    return {
        "emotion":              emotion,
        "confidence":           round(confidence, 4),
        "latency_ms":           latency_ms,
        "chunk_count":          chunk_count,
        "is_reliable":          is_reliable,
        "all_probs":            all_probs,
        "vad_silence_skipped":  round(silence_skipped_sec, 2),
        "vad_silence_pct":      round(silence_pct, 1),
        "vad_ms":               vad_ms,
        "vad_enabled":          vad_was_used,
        "realtime_factor":      round(speed_ratio, 2),
        "inference_ms":         round(infer_ms, 0),
        "avg_chunk_ms":         round(avg_chunk_ms, 0),
        "enabled":              config.EMOTION_ENABLED,
    }