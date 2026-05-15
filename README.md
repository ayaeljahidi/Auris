# Auris — High-Performance Multi-Modal Speech Analysis

Auris is a state-of-the-art speech analysis platform designed for real-time and batch processing. It combines high-speed transcription, contextual grammar correction, dual-source emotion fusion, and automated question generation into a single, optimized pipeline.

Built for efficiency on CPU-only environments, Auris utilizes a parallelized architecture to deliver comprehensive multi-modal insights with minimal latency.

---

## 🚀 Key Features

### 1. Advanced Transcription & Correction
*   **Whisper ASR:** Powered by `faster-whisper` for low-latency, quantized speech-to-text.
*   **Correction-as-a-Service:** Uses `Flan-T5` to monitor and correct grammar, spelling, and contextual nuances in real-time or batch modes.

### 2. Dual-Source Emotion Fusion
Auris doesn't just look at what is said, but *how* it is said. It merges two distinct emotional signals:
*   **Audio Emotion (Wav2Vec2):** Analyzes non-verbal cues (tone, pitch, energy) directly from the raw waveform.
*   **Text Emotion (DistilRoBERTa):** Performs semantic analysis on the transcript to detect emotional intent from the words.
*   **Adaptive Fusion Layer:** Intelligently weighs both signals, rewarding consensus and penalizing low-confidence disagreements.

### 3. Intelligent Question Generation
*   **Qwen-2.5 Integration:** Analyzes the final transcript via Ollama (Qwen-2.5:1.5b) to generate relevant follow-up questions or insights based on the conversation context.

### 4. Interactive Frontend
*   **Live Recording:** Real-time WebSocket-based audio capture with visual feedback.
*   **Analysis Dashboard:** Visual representation of transcripts, emotion scores, and generated questions with smooth GSAP animations and 3D terrain visuals.

---

## 🧠 Pipeline Architecture

The system operates using a **3-Phase Dependency Graph** to maximize hardware utilization:

1.  **Phase 1: Feature Extraction (T=0)**
    *   As soon as audio is received, **Whisper ASR** and **Audio Emotion** analysis launch concurrently.
2.  **Phase 2: Contextual Analysis**
    *   Triggered once the raw transcript is available. **Text Emotion** analysis and **Flan-T5 Correction** run in parallel.
3.  **Phase 3: Insight Generation**
    *   The **Emotion Fusion** layer merges the results, and **Qwen QG** generates follow-up questions.

---

## 🛠 Tech Stack

### Backend (Python/FastAPI)
*   **Framework:** FastAPI
*   **Models:** Faster-Whisper, Flan-T5, Wav2Vec2, DistilRoBERTa, Qwen-2.5 (via Ollama)
*   **Processing:** PyAV, NumPy, Torch, Transformers

### Frontend (React/TypeScript)
*   **Framework:** React 19 + Vite
*   **Styling:** Tailwind CSS + Shadcn/UI
*   **Visuals:** Three.js (@react-three/fiber), GSAP, Lenis (Smooth Scroll)

---

## 📦 Installation & Setup

### Prerequisites
*   Python 3.10+
*   Node.js (v18+)
*   [Ollama](https://ollama.com/) (for Question Generation)

### 1. Backend Setup
```bash
# Create and activate a virtual environment
python -m venv .venv
# Windows: .venv\Scripts\activate | Linux: source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Download and cache models (Whisper, Flan, Emotion models)
python scripts/setup_models.py

# Pull the Qwen model
ollama pull qwen2.5:1.5b
```

### 2. Frontend Setup
```bash
cd frontend
npm install
cd ..
```

---

## 🚀 Running the Application

To run the full stack, you will need two terminal windows:

**Terminal 1: Backend**
```bash
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

**Terminal 2: Frontend**
```bash
cd frontend
npm run dev
```
The application will be available at `http://localhost:3000`.

---

## ⚙️ Configuration

Environment variables can be set in `backend/config.py` or passed via CLI:

| Variable | Default | Description |
|---|---|---|
| `WHISPER_MODEL` | `base.en` | Whisper model size (tiny.en, base.en, small.en) |
| `FLAN_ENABLED` | `true` | Enable Flan-T5 grammar correction |
| `EMOTION_ENABLED` | `true` | Enable the dual-emotion pipeline |
| `EMOTION_AUDIO_WEIGHT`| `0.5` | Weight for audio-based emotion (0.0 to 1.0) |
| `EMOTION_TEXT_WEIGHT` | `0.5` | Weight for text-based emotion (0.0 to 1.0) |

---

## 📂 Project Structure

```text
Auris/
├── backend/            # FastAPI source code (audio, transcription, emotion, QG)
├── frontend/           # React frontend (Vite, Tailwind, Shadcn/UI)
│   ├── src/sections/   # Page components (LiveRecording, Upload, Results)
│   └── src/components/ # Reusable UI components
├── models/             # Local cache for AI models
├── scripts/            # Setup and utility scripts
├── requirements.txt    # Python dependencies
└── README.md           # This file
```
