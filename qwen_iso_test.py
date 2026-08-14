"""
Isolated Qwen test (diagnosis only -- no ASR, no TTS).

Feeds the EXACT strings the user observed through the production LLM (same
model, same sampling, same system prompt) with a fresh single-turn history
for each, and prints the RAW responses so we can judge whether Qwen itself
is failing or merely mirroring ASR-corrupted input.

String A : मशीन लर्निंग के बारे में बताएं।   (correct)
String B : मशीन लर्नी के बारे में बताएं।      (one phoneme dropped)
String C : मुशीन लर्रनिंके बारे में बताएं।     (heavily mangled)
String D : Tell me about machine learning.    (clean English control)
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from llm.local_llm import LocalLLM

STRINGS = {
    "A_correct": "मशीन लर्निंग के बारे में बताएं।",
    "B_dropped": "मशीन लर्नी के बारे में बताएं।",
    "C_mangled": "मुशीन लर्रनिंके बारे में बताएं।",
    "D_english": "Tell me about machine learning.",
}


def main():
    print("=" * 80)
    print("ISOLATED QWEN TEST (raw responses, single-turn, production prompt)")
    print("=" * 80)

    llm = LocalLLM()
    print("\n[1] Warming up model...")
    llm.warmup()

    results = {}
    for key, prompt in STRINGS.items():
        # fresh single-turn history: system prompt + this one user turn
        llm.reset_history()
        t0 = time.perf_counter()
        raw = llm.generate(prompt)
        latency = time.perf_counter() - t0
        prev = llm.history[-2]["content"] if len(llm.history) >= 2 else None
        results[key] = {
            "input": prompt,
            "raw_response": raw,
            "response_chars": len(raw),
            "latency_s": round(latency, 3),
            "history_user_msg_exact": prev,
        }
        print("\n" + "-" * 80)
        print(f"[{key}] input ({len(prompt)} ch): {prompt}")
        print(f"[{key}] raw response ({len(raw)} ch, {latency:.2f}s):")
        print(f"    {raw}")

    with open("qwen_iso_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("\nResults JSON -> qwen_iso_results.json")


if __name__ == "__main__":
    main()