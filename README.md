# AZMTH Voice Assistant

An end-to-end offline conversational AI assistant built with local speech
recognition, local language models, and local text-to-speech.

## Features

- Fully offline operation
- Real-time microphone input
- Voice Activity Detection (WebRTC VAD)
- Speech segmentation with silence-based utterance detection
- **Multilingual**: Whisper auto-detects the speaker's language each
  utterance and the assistant replies in that same language (EN + HI
  Piper voices installed; arbitrary languages are transcribed and mirrored
  where a matching voice exists, falling back to EN/HI)
- **Streaming pipeline**: the first complete sentence is spoken while the
  LLM keeps generating (Devanagari `।`/`॥` punctuation splits Hindi
  sentences too)
- **Barge-in**: the user can interrupt playback mid-response; the partial
  answer is preserved in conversation history
- **Garbage gate**: Whisper's hallucinated/silence output is rejected by an
  ASR validation layer (`asr/validation.py`) so the LLM and TTS only ever see
  plausible speech
- **Language stability**: detection confidence + last-language memory avoid
  random language flips; `AZMTH_LANG` pins a single language
- **Debug mode**: `AZMTH_DEBUG=1` dumps per-utterance WAVs to `debug_audio/`
  and prints ASR input/confidence/validation diagnostics
- Local speech-to-text using Whisper
- Local reasoning using GGUF language models
- Local text-to-speech using Piper
- **Built-in performance metrics**: every stage is timed and reported
  (see `METRICS.md`)
- No cloud APIs; privacy-first architecture

## Architecture

```
Microphone
   ↓
Audio Recorder (16 kHz, 20 ms blocks)
   ↓
WebRTC VAD  ⟶  Speech Segmenter (pre-roll 400 ms / tail 600 ms)
   ↓
Whisper ASR (config-driven: threads, gain, language detection + confidence)
   ↓
ASR Validation Gate  —  reject silence/hallucination/low-confidence
   ↓                        (debug block shows the reason)
Local LLM (Qwen2.5 1.5B GGUF, streaming tokens)
   ↓  (sentences split on . ! ? । ॥  as they complete)
Piper TTS (per-language voice, RTF ≈ 0.1)
   ↓
Speaker (OutputStream w/ barge-in, EOS→first-audio timing)
```

Everything tunable lives in `config.py` (env overrides, `AZMTH_*`),
including VAD timing, segment gates, ASR gain/threads/model, language
handling, and validation thresholds.

## Current Stack

- **Speech recognition**: OpenAI Whisper `small` (CPU, 16 kHz,
  `AZMTH_ASR_MODEL` env var selects any Whisper size, e.g. `base`/`medium`)
- **VAD**: WebRTC VAD, silence-based segmentation
- **LLM**: Qwen2.5 1.5B Instruct GGUF on llama.cpp (fully offline)
- **TTS**: Piper (en_US-lessac-medium, hi_IN-priyamvada-medium)

## Latency

The pipeline measures ASR time, LLM first-token / tokens/sec, TTS RTF,
queue depth, and end-to-end EOS→first-audio latency. Baseline numbers
and how they are collected are documented in **`METRICS.md`**; run
`python benchmark.py` any time to regenerate them.

## Quick start

```powershell
python main.py        # voice assistant; Ctrl+C prints metrics + writes CSV/JSON
python benchmark.py   # headless component + E2E latency benchmark
python asr_debug.py --repeats 3 --seconds 6   # record & diagnose ASR only (no LLM/TTS)
```

`asr_debug.py` is the first tool to reach for when ASR misbehaves: it
records, saves a WAV, runs language detection + transcription + the
validation gate, and prints a structured report.

## Project layout

- `asr/`     Whisper speech-to-text + `validation.py` garbage gate
- `vad/`     realtime + offline VAD segmentation
- `audio/`   recorder, output with barge-in
- `llm/`     local GGUF language model
- `tts/`     Piper voices and synthesis
- `pipeline/` pipeline controller + metrics collector
- `config.py` single source of tunables (`AZMTH_*` env overrides)
- `asr_debug.py` standalone ASR diagnostics tool
- `main.py`  threaded pipeline orchestration

AZMTH aims to build a privacy-first conversational AI that runs entirely
on edge devices, enabling intelligent voice interaction without reliance
on cloud infrastructure.