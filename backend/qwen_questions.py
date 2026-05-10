"""
backend/qwen_questions.py
Question generation via Ollama — qwen2.5:1.5b running locally with llama.cpp.
Fast CPU inference, no transformers needed.
"""

import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "qwen2.5:1.5b"
SYSTEM_PROMPT = """You are a jury member evaluating a student presentation.
Output ONLY 4 numbered questions. Each must be specific to the content below, one question mark only.
Format:
1. [question]
2. [question]
3. [question]
4. [question]"""



def _load_model() -> None:
    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=5)
        models = [m["name"] for m in r.json().get("models", [])]
        if not any("qwen2.5" in m for m in models):
            print(f"[Ollama] WARNING: {MODEL_NAME} not found. Run: ollama pull qwen2.5:3b")
            return

        # Send a tiny real prompt to force the model into RAM and wait for it
        print(f"[Ollama] Warming up {MODEL_NAME}...")
        requests.post(
            OLLAMA_URL,
            json={
                "model": MODEL_NAME,
                "prompt": "Hi",
                "stream": False,
                "keep_alive": -1,
                "options": {"num_predict": 1},  # Generate just 1 token — fast
            },
            timeout=120,  # Wait up to 2 min for cold load
        )
        print(f"[Ollama] ✓ {MODEL_NAME} is warm and ready")

    except Exception as e:
        print(f"[Ollama] WARNING: Warmup failed — {e}")
        
        
          
 
def generate_questions(transcribed_text: str) -> str:
    # Trim to 200 words — enough context for 4 specific questions
    words = transcribed_text.split()
    if len(words) > 200:
        transcribed_text = " ".join(words[:200]) + "…"

    prompt = f"{SYSTEM_PROMPT}\n\nPresentation:\n{transcribed_text}"

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL_NAME,
            "prompt": prompt,
            "stream": False,
            "keep_alive": -1,
            "options": {
                "temperature": 0.7,
                "top_p": 0.9,
                "repeat_penalty": 1.1,
                "num_predict": 180,
                "num_ctx": 512,   # minimal — prompt fits in ~350 tokens
            }
        },
        timeout=300,
    )

    response.raise_for_status()
    return response.json()["response"].strip()
