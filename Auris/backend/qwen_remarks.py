import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "qwen2.5:7b"
REMARKS_SYSTEM_PROMPT = """You are a jury member who just listened to a student group presentation.
Write exactly 5 remarks about the real problems you noticed in this speech.

Format — repeat exactly for each remark:
❌ [CATEGORY]: "[exact quote from the speech]"
[one simple sentence explaining why this is a problem]

Rules:
- exactly 5 remarks
- category must be one of: MEANING | FLOW | CLARITY | COHESION | LANGUAGE | STRUCTURE
- quote must be copied word for word from the transcript
- one sentence per remark, short and specific
- follow the order of the speech — first problem at top, last at bottom
- cover as many different categories as possible
- no intro, no conclusion, no extra text
- never use the words: lacks, vague, insufficient, potential, depth

Categories — hunt for these specific patterns:

MEANING: a claim with no number behind it / a result described with no metric / a sentence that sounds like information but says nothing

FLOW: a speaker switch with no transition sentence / a jump from one topic to another with no bridge / a conclusion that does not refer back to the problem stated at the start

CLARITY: a technical term dropped without explanation / a sentence too long to follow when spoken aloud

COHESION: a speaker says "I built" or "my idea" or "I designed" in a group project — this is the most important problem to catch / two speakers give contradictory information / the same component is called by two different names

LANGUAGE: filler words — basically, kind of, so, you know / a grammar error / an informal expression in front of a jury

STRUCTURE: the audience cannot tell what the project does in the first 30 seconds / no problem statement before the solution / the conclusion states no achievement"""

def generate_remarks(transcribed_text: str, model: str = MODEL_NAME) -> dict:
    prompt = f"{REMARKS_SYSTEM_PROMPT}\n\n{transcribed_text}"

    word_count = len(transcribed_text.split())
    ctx     = 1024 if word_count < 150 else 1536 if word_count < 400 else 2048
    predict = 300  if word_count < 150 else 380  if word_count < 400 else 450

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": model,
            "prompt": prompt,
            "stream": False,
            "keep_alive": -1,
            "options": {
                "temperature": 0.3,
                "top_p": 0.85,
                "repeat_penalty": 1.3,
                "num_predict": predict,
                "num_ctx": ctx,
            }
        },
        timeout=300,
    )
    response.raise_for_status()
    return response.json()["response"].strip()