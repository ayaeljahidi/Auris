"""
Flan-T5 Transcription Correction Tester — Custom Text
======================================================
Tests flan-t5-base and flan-t5-large on your real transcription.

Install dependencies first:
    pip install transformers torch sentencepiece accelerate
"""

from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch
import time
import textwrap

# ─────────────────────────────────────────────
# YOUR TRANSCRIPTION TEXT
# ─────────────────────────────────────────────

FULL_TEXT = """So the thing is when you want to build something great, it's not easy to do. \
And when you're doing something that's not easy to do, you're not always enjoying it. \
I don't love every day of my job. I don't think every day brings me joy, \
nor does joy have to be the definition of a good day. And every day I'm not happy. \
Every year I'm not happy about the company. But I love the company every single second. \
And so I think that what people misunderstand is somehow the best jobs are the one that brings \
you happiness all the time. I don't think that that's right. \
You have to suffer. You have to struggle. You have to endeavor. \
You have to do those hard things and work through it in order to really appreciate what \
you've done. And there are no such thing that are great that was easy to do. \
And so by definition, I would say, therefore, I wish upon you greatness, \
which by my way of saying it, I wish upon you plenty of plain and suffering. And so..."""

# Split into sentences for per-sentence correction
SENTENCES = [s.strip() for s in FULL_TEXT.replace("...", "").split(".") if s.strip()]

# ─────────────────────────────────────────────
# MODELS
# ─────────────────────────────────────────────

MODELS = {
    "flan-t5-base":  "google/flan-t5-base",   # ~250 MB
    "flan-t5-large": "google/flan-t5-large",  # ~800 MB
}

# ─────────────────────────────────────────────
# PROMPT TEMPLATES
# ─────────────────────────────────────────────

PROMPT_TEMPLATES = {
    "grammar fix":   lambda t: f"Fix the grammar in this sentence: {t}",
    "rewrite":       lambda t: f"Rewrite this with correct grammar and spelling: {t}",
    "transcription": lambda t: f"You are correcting a speech transcription. Fix any grammar, spelling, or word errors while keeping the original meaning: {t}",
}

# ─────────────────────────────────────────────
# GENERATION SETTINGS
# ─────────────────────────────────────────────

GEN_CONFIG = {
    "max_new_tokens": 300,
    "num_beams": 4,
    "early_stopping": True,
    "no_repeat_ngram_size": 3,
}

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def load_model(model_name, model_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n  📦 Loading {model_name} on {device.upper()}...")
    start = time.time()
    tokenizer = T5Tokenizer.from_pretrained(model_path)
    model = T5ForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    ).to(device)
    model.eval()
    print(f"  ✅ Loaded in {time.time()-start:.1f}s | Params: {model.num_parameters():,} | {device.upper()}")
    return model, tokenizer, device


def correct_sentence(text, prompt_fn, model, tokenizer, device):
    prompt = prompt_fn(text)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
    start = time.time()
    with torch.no_grad():
        outputs = model.generate(**inputs, **GEN_CONFIG)
    latency = (time.time() - start) * 1000
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return result, latency


def correct_full_text(sentences, prompt_fn, model, tokenizer, device):
    """Correct all sentences and join them back into a paragraph."""
    corrected_sentences = []
    total_latency = 0
    for sentence in sentences:
        corrected, latency = correct_sentence(sentence, prompt_fn, model, tokenizer, device)
        corrected_sentences.append(corrected)
        total_latency += latency
    full_corrected = ". ".join(corrected_sentences) + "."
    return full_corrected, total_latency


def print_diff(original, corrected):
    """Count how many words changed between original and corrected."""
    orig_words = original.lower().split()
    corr_words = corrected.lower().split()
    changes = 0
    for ow, cw in zip(orig_words, corr_words):
        if ow != cw:
            changes += 1
    return changes


def wrap(text, width=65, indent="       "):
    return textwrap.fill(text, width=width, subsequent_indent=indent)


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def run():
    print("=" * 70)
    print("  🧪 FLAN-T5 — TRANSCRIPTION CORRECTION TEST")
    print("=" * 70)
    print(f"  GPU available : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU           : {torch.cuda.get_device_name(0)}")
    print(f"  Sentences     : {len(SENTENCES)}")
    print()
    print("  📝 ORIGINAL TEXT:")
    print(f"  {wrap(FULL_TEXT)}")

    all_summaries = {}

    for model_name, model_path in MODELS.items():
        print(f"\n{'=' * 70}")
        print(f"  🤖 MODEL: {model_name.upper()}")
        print(f"{'=' * 70}")

        try:
            model, tokenizer, device = load_model(model_name, model_path)
        except Exception as e:
            print(f"  ❌ Could not load {model_name}: {e}")
            continue

        model_summaries = {}

        for prompt_name, prompt_fn in PROMPT_TEMPLATES.items():
            print(f"\n  ─── Prompt: [{prompt_name.upper()}] ───")
            print(f"  Template: \"{prompt_fn('...')}\"")
            print()

            # Sentence-by-sentence correction
            sentence_latencies = []
            for i, sentence in enumerate(SENTENCES, 1):
                corrected, latency = correct_sentence(sentence, prompt_fn, model, tokenizer, device)
                changes = print_diff(sentence, corrected)
                sentence_latencies.append(latency)

                print(f"  [{i:02d}] ⏱ {latency:.0f}ms | Changes: {changes} word(s)")
                print(f"       ❌ {wrap(sentence)}")
                print(f"       ✅ {wrap(corrected)}")
                print()

            # Full reassembled paragraph
            full_corrected, total_latency = correct_full_text(
                SENTENCES, prompt_fn, model, tokenizer, device
            )
            avg_latency = sum(sentence_latencies) / len(sentence_latencies)

            print(f"\n  📄 FULL CORRECTED TEXT [{prompt_name.upper()}]:")
            print(f"  {wrap(full_corrected)}")
            print(f"\n  ⚡ Total: {total_latency:.0f}ms | Avg/sentence: {avg_latency:.0f}ms")

            model_summaries[prompt_name] = {
                "avg_latency": avg_latency,
                "total_latency": total_latency,
                "full_text": full_corrected,
            }

        all_summaries[model_name] = model_summaries

        # Free memory before loading next model
        del model, tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ─────────────────────────────────────────────
    # FINAL COMPARISON
    # ─────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("  📊 FINAL COMPARISON")
    print(f"{'=' * 70}")
    print(f"\n  {'Model':<20} {'Prompt':<20} {'Avg/sentence':<15} {'Total'}")
    print(f"  {'-'*20} {'-'*20} {'-'*15} {'-'*12}")

    for model_name, prompts in all_summaries.items():
        for prompt_name, data in prompts.items():
            print(f"  {model_name:<20} {prompt_name:<20} "
                  f"{data['avg_latency']:.0f} ms          "
                  f"{data['total_latency']:.0f} ms")

    print(f"\n{'=' * 70}")
    print("  ✅ TEST COMPLETE!")
    print()
    print("  💡 KEY THINGS TO CHECK IN THE OUTPUT:")
    print()
    print("  1. 'plenty of plain and suffering'")
    print("     → Should be: 'plenty of pain and suffering'")
    print("     (homophone error — very common in transcriptions)")
    print()
    print("  2. 'the best jobs are the one that brings'")
    print("     → Should be: 'the best jobs are the ones that bring'")
    print("     (subject-verb agreement)")
    print()
    print("  3. 'there are no such thing that are great that was'")
    print("     → Should be: 'there are no such things that are great that were'")
    print("     (plural + tense agreement)")
    print()
    print("  4. 'I wish upon you plenty of plain and suffering'")
    print("     → The meaning/spirit of the speech must be PRESERVED")
    print("     (correction should not change the author's voice)")
    print(f"\n{'=' * 70}\n")


if __name__ == "__main__":
    run()