"""Auris — Speech Emotion Recognition using Wav2Vec2 (no compilation, 8 emotions)"""
import logging
import time
import torch
import librosa
import numpy as np
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor

from . import config

log = logging.getLogger("auris.emotion")

# Model name
MODEL_NAME = config.EMOTION_MODEL

# 8 emotion classes mapped from model's labels (ANG, CAL, DIS, FEA, HAP, NEU, SAD, SUR)
EMOTION_LABELS = ["angry", "calm", "disgust", "fear", "happy", "neutral", "sad", "surprised"]

# Singleton pattern - load model once
_emotion_model = None
_emotion_processor = None
_session_loaded = False


def _get_persistent_model():
    """Load Wav2Vec2 emotion model once (no compilation needed)."""
    global _emotion_model, _emotion_processor, _session_loaded
    
    if _session_loaded:
        return _emotion_model, _emotion_processor
    
    if not config.EMOTION_ENABLED:
        _session_loaded = True
        return None, None
    
    try:
        log.info(f"Loading speech emotion model: {MODEL_NAME}")
        log.info("This may take a moment on first run (downloading ~378MB model)...")
        
        _emotion_model = Wav2Vec2ForSequenceClassification.from_pretrained(MODEL_NAME)
        _emotion_model.eval()
        
        _emotion_processor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_NAME)
        
        _session_loaded = True
        
        num_labels = _emotion_model.config.num_labels
        log.info(f"✓ Emotion model loaded. {num_labels} classes: {EMOTION_LABELS}")
        
    except Exception as e:
        log.error(f"Failed to load emotion model: {e}")
        _session_loaded = True
        return None, None
    
    return _emotion_model, _emotion_processor


def detect_emotion_global(audio: np.ndarray, sr: int = 16000) -> dict:
    """
    Detect emotion from audio numpy array (float32, 16kHz recommended).
    
    Args:
        audio: numpy array of audio samples (float32, normalized to [-1, 1])
        sr: sample rate (will resample to 16kHz if needed)
    
    Returns:
        dict with emotion, confidence, and all class probabilities
    """
    t_start = time.perf_counter()
    
    not_available = {
        "emotion": "unknown",
        "confidence": 0.0,
        "latency_ms": 0,
        "chunk_count": 0,
        "is_reliable": False,
        "all_probs": {},
        "enabled": config.EMOTION_ENABLED,
        "model": None,
        "realtime_factor": 0,
        "inference_ms": 0,
    }
    
    model, processor = _get_persistent_model()
    
    if model is None or not config.EMOTION_ENABLED:
        return not_available
    
    # Ensure float32
    if audio.dtype != np.float32:
        audio = audio.astype(np.float32)
    
    # Normalize to [-1, 1] if needed (detect int16 range)
    if audio.size > 0 and np.abs(audio).max() > 1.0:
        audio = audio / 32768.0
    
    duration_sec = len(audio) / sr if sr > 0 else 0
    
    # Resample to 16kHz if needed (Wav2Vec2 expects 16kHz)
    if sr != 16000:
        log.debug(f"Resampling from {sr}Hz to 16000Hz")
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        sr = 16000
    
    # Limit to 30 seconds max (Wav2Vec2 memory constraint)
    max_samples = 30 * sr
    if len(audio) > max_samples:
        log.warning(f"Audio too long ({len(audio)/sr:.1f}s), truncating to 30s")
        audio = audio[:max_samples]
    
    try:
        # Process audio - returns input_values tensor
        inputs = processor(
            audio,
            sampling_rate=sr,
            return_tensors="pt",
            padding=True
        )
        
        # Run inference
        t_infer = time.perf_counter()
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probs = torch.nn.functional.softmax(logits, dim=1).squeeze()
        
        # Handle case where probs is 0-dim (single element)
        if probs.dim() == 0:
            probs = probs.unsqueeze(0)
        
        probs_list = probs.tolist()
        if isinstance(probs_list, float):
            probs_list = [probs_list]
        
        infer_ms = round((time.perf_counter() - t_infer) * 1000)
        
        # Get prediction
        pred_idx = int(torch.argmax(probs))
        emotion = EMOTION_LABELS[pred_idx]
        confidence = float(probs[pred_idx])
        is_reliable = confidence >= 0.5
        
        all_probs = {EMOTION_LABELS[i]: round(float(probs[i]), 4) 
                     for i in range(len(EMOTION_LABELS))}
        latency_ms = round((time.perf_counter() - t_start) * 1000)
        
        # Calculate speed factor
        speed_ratio = duration_sec / (latency_ms / 1000) if latency_ms > 0 else 0.0
        
        # Print nice output
        print()
        print("╔" + "═" * 60 + "╗")
        print("║" + " 🎭  SPEECH EMOTION (Wav2Vec2)".center(60) + "║")
        print("╠" + "═" * 60 + "╣")
        print(f"║  Emotion   : {emotion.upper():<15}  {'✓ RELIABLE' if is_reliable else '⚠ LOW CONFIDENCE'}".ljust(61) + "║")
        print(f"║  Confidence: {confidence:.1%}".ljust(61) + "║")
        print(f"║  Latency   : {latency_ms}ms  |  Inference: {infer_ms}ms".ljust(61) + "║")
        print(f"║  Duration  : {duration_sec:.1f}s  |  Speed: {speed_ratio:.1f}x realtime".ljust(61) + "║")
        print("╠" + "─" * 60 + "╣")
        for label, prob in sorted(all_probs.items(), key=lambda x: -x[1]):
            bar = "█" * int(prob * 20)
            print(f"║  {label:<10} {prob:>6.1%}  {bar}".ljust(61) + "║")
        print("╚" + "═" * 60 + "╝")
        
        log.info(f"✅ Emotion: {emotion} ({confidence:.1%}) | {latency_ms}ms | {speed_ratio:.1f}x realtime")
        
        return {
            "emotion": emotion,
            "confidence": round(confidence, 4),
            "latency_ms": latency_ms,
            "chunk_count": 1,
            "is_reliable": is_reliable,
            "all_probs": all_probs,
            "enabled": config.EMOTION_ENABLED,
            "model": MODEL_NAME,
            "duration_sec": round(duration_sec, 2),
            "realtime_factor": round(speed_ratio, 2),
            "inference_ms": infer_ms,
        }
        
    except Exception as e:
        log.error(f"Emotion detection failed: {e}")
        import traceback
        traceback.print_exc()
        return not_available


# Alias for backward compatibility
detect_emotion = detect_emotion_global