"""
backend/qwen_questions.py
Question generation via Ollama — qwen2.5:1.5b running locally with llama.cpp.
Fast CPU inference, no transformers needed.
"""

import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "qwen2.5:1.5b"
SYSTEM_PROMPT = """You are an expert jury member evaluating a technical student presentation.
Generate exactly 4 jury-style questions based on the transcribed content.

STRICT RULES:
- Each question must be ONE question only — do not combine multiple questions into one
- The question can be detailed and long but must have only ONE question mark at the end
- Do not use "and what", "also explain", "additionally" to chain multiple questions together
- Do NOT generate generic questions that could apply to any presentation
- Output ONLY the 4 numbered questions, nothing else

Format:
1. [one detailed question]
2. [one detailed question]
3. [one detailed question]
4. [one detailed question]"""



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
    prompt = f"{SYSTEM_PROMPT}\n\nHere is the transcribed presentation:\n\n{transcribed_text}"

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
            "num_predict": 250,   # ← reduced from 400, notes were eating most of those tokens
            "num_ctx": 4096, 
        }
        },
        timeout=300,
    )

    response.raise_for_status()
    return response.json()["response"].strip()
