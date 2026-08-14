# Azmth Voice Agent — Performance Metrics

This document defines every latency & throughput metric the voice pipeline
measures, how it is captured, and the baseline numbers recorded on the
reference machine. Use it as the source of truth when tuning or regressing.

## How metrics are collected

Two complementary mechanisms are used:

1. **Live collector** (`pipeline/metrics.py`). The running assistant
   (main.py) timestamps every stage per utterance and prints an aggregate
   report on exit. It also writes `metrics.csv` (per-utterance rows) and
   `metrics.json` (raw transactions + summary) to the working directory.

2. **Headless benchmark** (`benchmark.py`). Runs each stage and a full
   end-to-end utterance without a microphone, producing repeatable
   component numbers and `benchmark_results.json`. It excludes silence
   detection and audio playback because no sound hardware is involved.

### Metric definitions

| Metric | Definition | Where it is measured |
|---|---|---|
| **ASR time** | Wall time from a finished utterance segment to its transcript (includes automatic language detection). | `WhisperASR.transcribe` wrapper in `asr_worker_loop`. |
| **LLM first-token latency** | Time from end-of-speech to the first generated token. Because it is measured from EOS, it also includes queue wait + ASR time. See **audio→first-token**. | `llm_worker_loop`. |
| **LLM tokens/sec** | `tokens / time_after_first_token` (excludes prompt processing). | `llm_worker_loop`. |
| **TTS synthesis / sentence** | Wall time to synthesize a single sentence. | `tts_worker_loop`. |
| **TTS real-time factor (RTF)** | `synthesis_time / audio_duration`. < 1 means faster than realtime. | `benchmark.py`, live collector. |
| **EOS→first-audio (response latency)** | Time from end-of-speech (segment emitted) to the first sample reaching the output sound card. This is the number the user actually feels. | `metrics.record_first_audio` anchored by the `AudioOutput.on_first_audio` hook. |
| **Total E2E (serialized)** | Time from audio ready → ASR → full LLM response → full TTS, computed sequentially. Worst case; the live pipeline streams TTS sentence-by-sentence. | `benchmark.py` E2E runs. |
| **Queue depth** | Samples of `asr/llm/tts` queue sizes once per second; indicates backpressure. | `queue_sampler_loop`. |

## Baseline results (reference machine)

- OS: Windows 11 (10.0.26200), Python 3.10.4, **18 logical CPUs**
- Stack: Whisper `small` · Qwen2.5 **1.5B** Instruct Q4_K_M · Piper `medium`
  (en_US-lessac, hi_IN-priyamvada) · 16 kHz audio
- ASR input audio: 1.6–2.0 s synthesized utterances

### Per-stage (after model warm-up)

Numbers below are for the **app path**: explicit `detect_language` +
transcribe with that language (`threads=8`, gain 1.5) — what the live
assistant actually does. 8 threads measured ~1.6× faster than 4.

| Stage | Metric | Value |
|---|---|---|
| ASR (small) | transcription of 1.8 s EN (explicit lang) | **2.3 s** (RTF ≈ 1.3×) |
| ASR (small) | language detect (encoder pass) | **2.0 s** |
| ASR (small) | full app path (detect + transcribe, EN) | **≈ 4.6 s** |
| ASR (small) | full app path (HI) | **≈ 5.6 s** |
| ASR (small) | cold first transcription (autodetect) | 5.2 s |
| ASR (base)  | transcription, same utterance | **2.3 s** — lower Hindi accuracy (leaks Arabic script) |
| LLM | first token (median) | **0.14 s** |
| LLM | throughput (warm) | **≈ 21 tokens/s** |
| LLM | warm-up time | 0.5 s |
| TTS EN | synthesis RTF | **0.07–0.09** (≈ 12× realtime) |
| TTS HI | synthesis RTF | **0.08–0.14** (≈ 8× realtime) |
| VAD | per 20 ms chunk | < 2 ms |

### End-to-end (serialized worst case; includes language detection)

| Utterance | ASR | LLM first token | LLM total | TTS | **Total** |
|---|---|---|---|---|---|
| EN: "What is the capital of France?" | 4.64 s | 0.15 s | 0.49 s | 0.19 s | **5.3 s** |
| HI: "फ्रांस की राजधानी क्या है?" | 5.59 s | 0.35 s | 2.16 s | 0.27 s | **8.0 s** |

### Where the time goes

- **ASR is ~70–90 % of response latency.** Everything else is sub-second
  after warm-up. Reducing ASR time is the single highest-leverage change.
  `AZMTH_ASR_MODEL=base` trades ~3× ASR speed for worse multilingual
  accuracy (it can leak Arabic script into Hindi) — use it only for
  English-first deployments. `AZMTH_LANG=en` (or `hi`) skips language
  detection entirely and cuts total ASR to the bare transcribe (~2.3 s).
- **Hindi sentence streaming.** Devanagari sentences are now split on
  danda (`।`,`॥`) so the first Hindi sentence reaches TTS immediately
  instead of waiting for the whole response. Combined with a stricter
  system prompt + sampling (see below), Hindi E2E dropped from **15.8 s →
  ~8 s** (the 1.5B model previously rambled to ~88 tokens; it now
  answers in one or two punctuated sentences).
- **ASR validation gate makes garbage never reach the LLM.** Whisper
  `small` confidently hallucinates on silence/noise (`vous`, `ლლლ…`,
  English-forced output). A gate now rejects transcripts whose
  `no_speech_prob`, `compression_ratio`, `avg_logprob`, token-repetition,
  or length fail thresholds, so Qwen and Piper only ever see plausible
  speech (see `asr/validation.py`).

### Latency budget for the live pipeline

`response latency ≈ VAD end-of-speech (0.6 s silence + 0.4 s pre-roll)
+ queue wait (~0) + ASR (4.6–5.6 s) + LLM first token (0.15–0.35 s)
+ first TTS synthesis (0.1–0.3 s)` ≈ **5.5–6.5 s** for a one-sentence
English answer (≈ 8 s for Hindi, dominated by the weaker 1.5B LLM).
Length of the LLM response barely matters because TTS streams
sentence-by-sentence.

## Tuning knobs

| Knob | Effect | Trade-off |
|---|---|---|
| `AZMTH_ASR_MODEL=base` | ASR → ~2.3 s | Lower Hindi accuracy |
| `AZMTH_ASR_MODEL=medium` | Better multilingual accuracy | Higher ASR latency |
| `AZMTH_ASR_THREADS=8` | Verified ~1.6× faster than 4 (3.2 s vs 5.9 s on quiet speech) | Default; >8 no better |
| `AZMTH_ASR_GAIN=1.5` | Amplifies quiet mics so Whisper sees louder input | Noise also amplified |
| `AZMTH_LANG=en` / `hi` | Skip detection, −2 s per utterance | Fixed language |
| `AZMTH_LANG_CONF=0.5` | Detection confidence floor; below it, previous language / `AZMTH_DEFAULT_LANG` is kept | Too low → wrong language, too high → sticky |
| `AZMTH_MIN_SEGMENT_PEAK=3000` | Drops segments below this int16 peak | Raised → ignores quiet/noise |
| `NO_SPEECH_THRESHOLD / MAX_COMPRESSION_RATIO / MIN_AVG_LOGPROB / MAX_TOKEN_REPEAT_FRAC` | ASR gate thresholds | Tune per mic noise level |
| `AZMTH_DEBUG=1` | Writes debug WAVs to `debug_audio/` + per-utterance ASR diagnostics | Slower; disk churn |
| Reduce `n_ctx` in `LocalLLM` | Less KV-cache memory | None for short convos |
| Whisper `without_timestamps`, no prev-text conditioning | ~10 % ASR speed-up, fewer hallucinations | No word timestamps (unneeded) |
| LLM `temperature=0.6, top_p=0.9, repeat_penalty=1.15` | Stable, concise multilingual answers | Less creative responses |

## Known issues & fixes

- **First-audio hook crashing the audio thread.** The PortAudio callback passes a
  `PaStreamCallbackTimeInfo` object as its `time` argument, which shadowed the
  `time` module and raised `AttributeError: ... no field 'perf_counter'`. Fixed by
  renaming the callback parameter to `_time_info`; the hook is also wrapped in
  `try/except` so a metrics failure can never interrupt playback. The active model
  name is printed at startup (`AZMTH_ASR_MODEL`).
- **Hallucinated/stray ASR reaching the LLM/TTS.** Previously any Whisper output —
  including silence ("vous") and noise ("ლლლ…") — was sent straight to Qwen and
  Piper. Fixed with `asr/validation.py`: on rejection the utterance is dropped
  (`asr_rejected` counter, debug block shows the reason). Verified live: near-silent
  mic → no_speech 0.857 → REJECTED; the LLM never fires.
- **Hindi ASR accuracy.** Whisper `small` mis-transcribes noisy/repeated Hindi
  (e.g. "मशीन लर्निंग" → "मशिल लरनी"), and sometimes prepends a spurious "अप"
  and mangles "राजधानी" even on clean TTS audio. Accepted as a model limit;
  `medium` or faster-whisper improves it. The 1.5B LLM can also hallucinate Hindi
  facts — see the accuracy limits below.
- **Language flips on garbage.** Detection confidence was near-random on
  silence/noise. `resolve_language` now remembers the last confident language and
  only switches when the new confidence exceeds `AZMTH_LANG_CONF`.

## Re-running

```powershell
python benchmark.py          # headless per-stage + E2E numbers
python main.py               # live assistant; Ctrl+C prints aggregate + writes metrics.csv/.json
```

Expected accuracy limits: the 1.5B LLM can hallucinate facts in Hindi
(e.g. answered the France question incorrectly); a 3B/7B GGUF improves
accuracy at higher latency. Piper voices currently cover EN + HI only.