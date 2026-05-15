# Auris Backend — Optimized Multi-Modal Pipeline

Auris is a high-performance FastAPI application designed for real-time speech-to-text, emotion detection, and automated question generation. It utilizes a highly parallelized architecture to deliver multi-modal insights with minimal latency.

## 🚀 Pipeline Architecture

The system operates using a **3-Phase Dependency Graph** to maximize CPU utilization and minimize the "Time-to-Result."

### Phase 1: Feature Extraction (T=0)
As soon as audio data is received, the following processes launch concurrently:
*   **Whisper ASR:** Transcribes audio to raw text.
*   **Audio Emotion (Wav2Vec2):** Analyzes tone, pitch, and energy from the raw waveform.
*   **Streaming Flan-T5:** A "Correction-as-a-Service" layer that monitors Whisper's output. It corrects grammar and context for each segment *as it is emitted*, rather than waiting for the full recording to finish.

### Phase 2: Contextual Analysis
Triggered as soon as the final corrected transcript is ready:
*   **Text Emotion (DistilRoBERTa):** Performs semantic analysis on the transcript to detect emotional intent from the words.
*   **Qwen QG (Question Generation):** Analyzed the transcript via Ollama (Qwen-2.5) to generate follow-up questions.

### Phase 3: Emotion Fusion
*   **Consensus Layer:** Merges Audio and Text emotion signals using adaptive weighting. It rewards consensus and penalizes low-confidence disagreements to provide a single, reliable emotional state.

---

## 🧠 Model Stack

| Task | Model | Specialized For |
| :--- | :--- | :--- |
| **ASR** | `faster-whisper` | Low-latency, quantized speech-to-text. |
| **Correction** | `Flan-T5` | Grammar, spelling, and contextual refinement. |
| **Audio Emotion** | `Wav2Vec2` | Non-verbal emotional cues (prosody). |
| **Text Emotion** | `DistilRoBERTa` | Lexical emotional intent. |
| **Question Gen** | `Qwen-2.5` | Reasoning and logic for follow-up Q&A. |

---

## ⚡ Key Optimizations

Auris is engineered for speed, especially in CPU-only environments.

### 1. Architectural Optimizations
*   **Multi-Phase Parallelism:** Uses `ThreadPoolExecutor` to execute independent model tasks simultaneously.
*   **Streaming Inference:** The Flan-T5 correction layer operates on a per-segment basis, significantly reducing perceived latency for long audio files.
*   **Concurrent Loading:** All models are pre-loaded in parallel during server startup, reducing cold-start time by up to 70%.

### 2. Computational Optimizations
*   **Quantization:** Uses `int8` quantization for Whisper, providing near-original accuracy with 4x faster inference on CPUs.
*   **Zero-Copy Memory:** Audio processing uses `numpy` views and pre-allocated buffers to prevent unnecessary memory copies and Garbage Collection (GC) pauses.
*   **Thread Tuning:** Torch thread pools are explicitly configured to match the hardware core count, preventing "thread thrashing."
*   **Ollama Offloading:** Offloads heavy LLM tasks (Question Generation) to an external process via async HTTP, keeping the main process responsive for real-time audio.

### 3. Logic & Reliability
*   **Adaptive Fusion:** The system dynamically adjusts weights between Audio and Text emotion based on word count and model confidence.
*   **Quality Gates:** Critique thresholds (logprobs/compression ratios) determine if a segment actually needs LLM correction, skipping unnecessary work for high-quality audio.

---

## 🛠 Setup & Configuration

Configuration is managed via environment variables in `config.py`.

*   `WHISPER_MODEL`: Default `base.en`
*   `FLAN_ENABLED`: Set to `true` for auto-correction.
*   `EMOTION_ENABLED`: Toggles the multi-modal emotion pipeline.
*   `TORCH_COMPILE`: Set to `true` on supported systems for kernel-level optimization.
