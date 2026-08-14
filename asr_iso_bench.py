"""
Isolated ASR benchmark (diagnosis only -- no LLM, no TTS, no VAD).

Captures the RAW microphone stream (exact bytes the pipeline sees before VAD),
saves every WAV, re-runs the CURRENT openai-whisper implementation AND
faster-whisper (same model size) on the SAME file, and prints a full audio +
inference report. Optionally compares against a clean Piper-synthesized
reference of the same sentence to separate room/audio degradation from a
Whisper model weakness.

Usage:
  python asr_iso_bench.py --repeats 3 --seconds 8
  python asr_iso_bench.py --wav path\to\capture.wav
  python asr_iso_bench.py --no-record
"""
import argparse
import json
import os
import sys
import time
import wave

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from asr.whisper_asr import WhisperASR

OUT_DIR = "asr_iso_audio"
os.makedirs(OUT_DIR, exist_ok=True)

GOLDEN_SENTENCES = {
    "machine_learning_hi": "मशीन लर्निंग के बारे में बताएं।",
    "what_is_ml_hi": "मशीन लर्निंग क्या है",
}


def save_raw_wav(samples_int16, path, rate=config.SAMPLE_RATE):
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(np.asarray(samples_int16, dtype="<i2").tobytes())


def load_raw_wav(path):
    with wave.open(path, "rb") as wf:
        rate = wf.getframerate()
        channels = wf.getnchannels()
        width = wf.getsampwidth()
        n = wf.getnframes()
        raw = wf.readframes(n)
    samples = np.frombuffer(raw, dtype="<i2")
    if channels > 1:
        samples = samples.reshape(-1, channels)[:, 0]
    return samples.astype(np.float32) / 32768.0, {
        "sample_rate": rate,
        "channels": channels,
        "sampwidth_bytes": width,
        "frames": n,
    }


def audio_meta(samples_f32, rate):
    """RMS / peak / duration / dtype report for a float32 signal."""
    return {
        "duration_s": round(len(samples_f32) / rate, 3),
        "rms_f32": float(np.sqrt(np.mean(samples_f32 ** 2))),
        "peak_f32": float(np.max(np.abs(samples_f32))),
        "peak_int16": int(np.max(np.abs(samples_f32 * 32768.0))),
        "dtype_convert": "float32 (from int16) -> [-1, 1]",
    }


def record_capture(path, seconds):
    """Record the raw int16 16 kHz mono mic stream with a live level meter.
    No VAD, no segmentation, no gain -- the exact bytes the pipeline sees."""
    import sounddevice as sd

    blocks = []
    meters = []

    def cb(indata, frames, _t, status):
        blocks.append(indata.flatten().copy())
        meters.append(int(np.max(np.abs(indata))))

    stream = sd.InputStream(
        samplerate=config.SAMPLE_RATE,
        channels=1,
        dtype="int16",
        blocksize=config.BLOCKSIZE,
        callback=cb,
    )
    stream.start()
    t0 = time.time()
    try:
        while time.time() - t0 < seconds:
            peak = max(meters[-20:]) if meters else 0
            bar = "#" * min(50, peak // (32768 // 50))
            print(f"\r  [rec {seconds:.0f}s] level: peak={peak:5d} |{bar:<50}|", end="")
            time.sleep(0.05)
    finally:
        stream.stop()
        stream.close()
    print()
    samples = np.concatenate(blocks) if blocks else np.array([], dtype=np.int16)
    save_raw_wav(samples, path)
    return samples


def piper_golden(sentence, path):
    """Synthesize a clean reference for the same sentence, saved as int16 16k mono."""
    from tts.piper_tts import PiperTTS

    tts_hi = PiperTTS("tts/hi_IN-priyamvada-medium.onnx")
    f32_chunks = list(tts_hi.synthesize_chunks(sentence))
    audio = np.concatenate([c for c, _ in f32_chunks])
    sr = f32_chunks[0][1] if f32_chunks else 22050
    if sr != config.SAMPLE_RATE:
        x = np.linspace(0, 1, len(audio))
        xp = np.linspace(0, 1, int(len(audio) * config.SAMPLE_RATE / sr))
        audio = np.interp(xp, x, audio)
    samples16 = np.clip(audio, -1, 1) * 32767
    save_raw_wav(samples16.astype(np.int16), path)
    return samples16.astype(np.int16)


def run_openai_whisper(whisper_asr, samples_f32, resolved_lang):
    """The CURRENT production implementation, on the exact supplied audio."""
    import torch

    # ctranslate2 (imported for faster-whisper) resets torch's OpenMP thread
    # count process-wide; re-assert so latency is comparable to the app.
    torch.set_num_threads(config.ASR_THREADS)
    t0 = time.perf_counter()
    detected, confidence = whisper_asr.detect_language(samples_f32)
    text, lang, stats = whisper_asr.transcribe(
        samples_f32, language=resolved_lang, return_stats=True)
    latency = time.perf_counter() - t0
    return {
        "backend": "openai-whisper (current app code)",
        "device": str(whisper_asr.model.device),
        "compute_type": "fp32 (fp16 unsupported on CPU)",
        "threads": config.ASR_THREADS,
        "model": whisper_asr.model_name,
        "detected_language": detected,
        "detected_confidence": round(confidence, 4),
        "resolved_language": resolved_lang,
        "transcript": text,
        "latency_s": round(latency, 3),
        "no_speech_prob": stats.get("no_speech_prob"),
        "avg_logprob": stats.get("avg_logprob"),
        "compression_ratio": stats.get("compression_ratio"),
        "token_count": len(stats.get("tokens") or []),
        "torch_num_threads": torch.get_num_threads(),
    }


def run_faster_whisper(faster_model, samples_f32, resolved_lang):
    """faster-whisper (ctranslate2 int8), same model size, on the SAME audio.
    No VAD (vad_filter=False) -- whole file transcribed as-is."""
    t0 = time.perf_counter()
    segments, info = faster_model.transcribe(
        samples_f32,
        language=resolved_lang,
        beam_size=1,
        temperature=0.0,
        vad_filter=False,
    )
    segs = list(segments)
    latency = time.perf_counter() - t0
    text = "".join(s.text for s in segs).strip()
    first = segs[0]
    return {
        "backend": "faster-whisper (ctranslate2)",
        "device": "cpu",
        "compute_type": "int8",
        "model": "small",
        "detected_language": info.language,
        "detected_confidence": round(info.language_probability, 4),
        "resolved_language": resolved_lang,
        "transcript": text,
        "latency_s": round(latency, 3),
        "no_speech_prob": getattr(first, "no_speech_prob", None),
        "avg_logprob": getattr(first, "avg_logprob", None),
        "compression_ratio": getattr(first, "compression_ratio", None),
        "token_count": len(getattr(first, "tokens", []) or []),
        "n_segments": len(segs),
    }


def analyze_one(whisper_asr, faster_model, f32, resolved_lang="hi"):
    meta = {"resolution": "RAW MIC / NO VAD / NO GAIN" if True else None}
    return {
        "audio_meta": audio_meta(f32, config.SAMPLE_RATE),
        "openai_whisper": run_openai_whisper(whisper_asr, f32, resolved_lang),
        "faster_whisper": run_faster_whisper(faster_model, f32, resolved_lang),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=3, help="live captures to record")
    ap.add_argument("--seconds", type=int, default=8, help="seconds per capture")
    ap.add_argument("--wav", action="append", default=[], help="analyze existing WAVs")
    ap.add_argument("--no-record", action="store_true", help="skip live recording")
    ap.add_argument("--language", default="hi", help="resolved language for transcription")
    args = ap.parse_args()

    print("=" * 80)
    print("ISOLATED ASR BENCHMARK (no LLM, no TTS, no VAD)")
    print("=" * 80)

    print("\n[1] Loading openai-whisper + faster-whisper (small)...")
    whisper_asr = WhisperASR(model_name="small", threads=config.ASR_THREADS, gain=1.0)
    t0 = time.perf_counter()
    from faster_whisper import WhisperModel
    faster_model = WhisperModel("small", device="cpu", compute_type="int8")
    print(f"    loaded in {time.perf_counter() - t0:.1f}s")

    corpora = {}

    corpora = {
        # WAVs recorded in an earlier run are still analyzed (existence is guarded later).
        "live_captures": [os.path.join(OUT_DIR, f"live_rec_{i + 1:02d}.wav") for i in range(args.repeats)],
        "existing_wavs": list(args.wav),
    }
    if not args.no_record and args.repeats > 0:
        print(f"\n[2] Recording {args.repeats} RAW captures ({args.seconds}s each) -- speak the same phrase repeatedly.")
        for i in range(args.repeats):
            p = os.path.join(OUT_DIR, f"live_rec_{i + 1:02d}.wav")
            record_capture(p, args.seconds)
            print(f"    saved RAW WAV -> {p}")

    print("\n[3] Synthesizing clean Piper references (same sentences, no room/audio effects)...")
    golden_paths = []
    for key, sentence in GOLDEN_SENTENCES.items():
        p = os.path.join(OUT_DIR, f"golden_{key}.wav")
        piper_golden(sentence, p)
        golden_paths.append(p)
        print(f"    saved -> {p}  ({sentence})")
    corpora["golden"] = golden_paths

    results = {}
    for group, paths in corpora.items():
        for p in paths:
            if not os.path.exists(p):
                print(f"    skip (missing): {p}")
                continue
            samples16, wavmeta = load_raw_wav(p)
            f32 = samples16.astype(np.float32)
            r = analyze_one(whisper_asr, faster_model, f32, args.language)
            # exact capture file metadata (bytes on disk)
            r["wav_meta_on_disk"] = wavmeta
            results[p] = r

    out_json = os.path.join(OUT_DIR, "asr_iso_results.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 80)
    print("REPORT")
    print("=" * 80)
    for p, r in results.items():
        print(f"\n>>> {p}")
        m = r["wav_meta_on_disk"]
        am = r["audio_meta"]
        print(f"  [File ] {m['sample_rate']} Hz, {m['channels']} ch, {m['sampwidth_bytes']}-byte, "
              f"{am['duration_s']}s, rms={am['rms_f32']:.4f}, peak={am['peak_f32']:.4f} "
              f"(int16 {am['peak_int16']}), dtype=int16->float32")
        for backend_key in ("openai_whisper", "faster_whisper"):
            b = r[backend_key]
            print(f"  [{b['backend']}]")
            print(f"    model={b['model']} device={b['device']} compute={b['compute_type']}")
            print(f"    lang_detect={b['detected_language']} (conf {b['detected_confidence']}) "
                  f"resolved={b['resolved_language']}")
            print(f"    transcript = '{b['transcript']}'")
            print(f"    latency={b['latency_s']}s  no_speech={b['no_speech_prob']} "
                  f"avg_logprob={b['avg_logprob']} comp_ratio={b['compression_ratio']} "
                  f"tokens={b['token_count']}")

    print(f"\nResults JSON -> {out_json}")


if __name__ == "__main__":
    main()