"""
Purpose
-------
Isolated, headless ASR test tool: record audio, save a WAV, run Whisper, and
print language confidence, transcript, latency and validation outcome. Works
completely independently of the LLM and TTS so the speech-to-text stage can be
benchmarked and debugged on its own.

Usage
-----
    python asr_debug.py --repeats 3 --seconds 6
    python asr_debug.py --language hi --repeats 2 --seconds 5
    python asr_debug.py --wav path/to/utterance.wav   (analyze an existing file)

Options mirror config.py; environment variables (AZMTH_ASR_MODEL etc.) also
apply. Use AZMTH_DEBUG=1 to get the full structured diagnostic block too.
"""

import argparse
import os
import sys
import time
import wave

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")

import sounddevice as sd

import config
from asr.whisper_asr import WhisperASR
from asr.validation import ASRValidation


def record(seconds, sample_rate, save_path):
    frames = []
    def cb(indata, frames_count, time_info, status):
        frames.append(indata.flatten().copy())
    stream = sd.InputStream(samplerate=sample_rate, channels=1, dtype="int16", callback=cb)
    print(f"Recording {seconds}s... speak now.")
    stream.start()
    time.sleep(seconds)
    stream.stop()
    stream.close()
    audio = np.concatenate(frames) if frames else np.zeros(0, dtype=np.int16)
    with wave.open(save_path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio.tobytes())
    print(f"Saved {save_path} ({len(audio) / sample_rate:.2f}s)")
    return audio.astype(np.int16)


def load_wav(path):
    with wave.open(path, "rb") as wf:
        assert wf.getsampwidth() == 2, "expected 16-bit PCM"
        data = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)
    return data


def analyze(asr, validator, audio, forced_language, label, debug_dir, sample_rate):
    orig = audio
    if len(orig) == 0:
        print(f"[{label}] empty capture, skipping")
        return

    duration = len(orig) / sample_rate
    peak = int(np.max(np.abs(orig)))
    rms = float(np.sqrt(np.mean(orig.astype(np.float32) ** 2)))

    # language detection (unless forced)
    if forced_language:
        detected, conf = forced_language, 1.0
        trusted = True
    else:
        detected, conf = asr.detect_language(orig)
        trusted = conf >= config.LANG_CONF_THRESHOLD

    t0 = time.perf_counter()
    text, out_lang, stats = asr.transcribe(orig, language=detected, return_stats=True)
    asr_time = time.perf_counter() - t0
    stats["asr_time"] = asr_time

    accepted, reason = validator.validate(
        text,
        no_speech_prob=stats["no_speech_prob"],
        avg_logprob=stats["avg_logprob"],
        compression_ratio=stats["compression_ratio"],
        tokens=stats["tokens"],
    )

    print(f"\n=== {label} ===")
    print(f"[Audio] duration={duration:.2f}s peak={peak} rms={rms:.0f} (of 32767)")
    print(f"[ASR]   model={asr.model_name} device=cpu lang={out_lang} "
          f"lang_conf={conf:.3f} {'(forced)' if forced_language else ('trusted' if trusted else 'carried-over')}")
    print(f"[ASR]   time={asr_time:.2f}s transcript={text!r}")
    print(f"[ASR]   no_speech_prob={stats['no_speech_prob']:.3f} "
          f"avg_logprob={stats['avg_logprob']:.3f} compression_ratio={stats['compression_ratio']:.2f} "
          f"tokens={len(stats['tokens'] or [])}")
    print(f"[Validation] {'ACCEPTED' if accepted else 'REJECTED'} reason={reason}")

    if debug_dir:
        os.makedirs(debug_dir, exist_ok=True)
        path = os.path.join(debug_dir, label.replace("/", "_") + ".wav")
        orig = np.clip(orig.astype(np.float32) * asr.gain, -32768, 32767).astype(np.int16)
        with wave.open(path, "wb") as wf:
            wf.setnchannels(1); wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(orig.tobytes())
        print(f"[Saved as] {path}")
    return accepted, reason, asr_time


def main():
    ap = argparse.ArgumentParser(description="Isolated ASR test/benchmark")
    ap.add_argument("--repeats", type=int, default=2, help="number of recordings")
    ap.add_argument("--seconds", type=float, default=5.0, help="seconds per recording")
    ap.add_argument("--language", default=None, choices=["en", "hi", "mr", None],
                    help="force a language (default: auto-detect)")
    ap.add_argument("--model", default=None, help="override Whisper model (default: config)")
    ap.add_argument("--wav", default=None, help="analyze an existing WAV instead of recording")
    ap.add_argument("--save-dir", default=None, help="folder to save recordings")
    args = ap.parse_args()

    asr = WhisperASR(args.model)
    validator = ASRValidation(config)
    save_dir = args.save_dir or "asr_debug_audio"

    labels = []
    if args.wav:
        audio = load_wav(args.wav)
        analyze(asr, validator, audio, args.language, f"file:{os.path.basename(args.wav)}", save_dir, config.SAMPLE_RATE)
    else:
        for i in range(args.repeats):
            save_path = os.path.join(save_dir, f"rec_{i + 1:03d}.wav")
            os.makedirs(save_dir, exist_ok=True)
            audio = record(args.seconds, config.SAMPLE_RATE, save_path)
            analyze(asr, validator, audio, args.language, f"rec_{i + 1:03d}", save_dir, config.SAMPLE_RATE)

    print("\nDone.")


if __name__ == "__main__":
    main()