"""
Purpose
-------
Single source of truth for every tunable in the voice pipeline.

Rationale
---------
Constants were previously scattered across main.py, the ASR wrapper and the
VAD segmenter with no way to tune behaviour at runtime. Centralizing them here
(with environment-variable overrides) makes the system adjustable on edge
deployments without touching code, and keeps behaviour consistent across the
live assistant, the benchmark and the ASR debug tool.

Assumptions
-----------
- All values have safe defaults that work on a quiet laptop mic + CPU-only box.
- Env vars are the override mechanism: AZMTH_* , documented in README/METRICS.
- Nothing here imports AI libraries, so config is cheap to load anywhere.
"""

import os


def _env(name, default):
    value = os.environ.get(name)
    return default if value is None or value == "" else value


def _env_int(name, default):
    value = _env(name, default)
    return int(value)


def _env_float(name, default):
    value = _env(name, default)
    return float(value)


# --------------------------------------------------------------------------- #
# Audio capture
# --------------------------------------------------------------------------- #
SAMPLE_RATE = 16000
BLOCKSIZE = 320          # 20 ms frames for VAD
OUTPUT_BLOCKSIZE = 512   # playback stream block size

# --------------------------------------------------------------------------- #
# Barge-in (dynamic echo-suppression threshold)
# --------------------------------------------------------------------------- #
BARGE_IN_PEAK = _env_int("AZMTH_BARGE_IN_PEAK", 4000)
BARGE_IN_FRAMES = _env_int("AZMTH_BARGE_IN_FRAMES", 5)

# --------------------------------------------------------------------------- #
# VAD / segmentation
# --------------------------------------------------------------------------- #
VAD_AGGRESSIVENESS = _env_int("AZMTH_VAD_AGGRESSIVENESS", 3)
PRE_SPEECH_MS = _env_int("AZMTH_PRE_SPEECH_MS", 400)        # padding before speech onset
SILENCE_TAIL_MS = _env_int("AZMTH_SILENCE_TAIL_MS", 600)    # trailing silence before cut
MIN_SEGMENT_DURATION = _env_float("AZMTH_MIN_SEGMENT_DURATION", 0.5)
MAX_SEGMENT_DURATION = _env_float("AZMTH_MAX_SEGMENT_DURATION", 8.0)
MIN_SEGMENT_PEAK = _env_int("AZMTH_MIN_SEGMENT_PEAK", 3000)  # int16 amplitude sanity gate

# --------------------------------------------------------------------------- #
# ASR (Whisper)
# --------------------------------------------------------------------------- #
ASR_MODEL = _env("AZMTH_ASR_MODEL", "small")
ASR_THREADS = _env_int("AZMTH_ASR_THREADS", 8)   # 8 measured ~1.5x faster than 4 on 18 cores
ASR_GAIN = _env_float("AZMTH_ASR_GAIN", 1.5)     # linear pre-gain applied to quiet mic audio
LANG_CONF_THRESHOLD = _env_float("AZMTH_LANG_CONF", 0.5)  # min top-1 language confidence
DEFAULT_LANG = _env("AZMTH_LANG", None)          # force a language for TTS/ASR (e.g. "hi")

# --------------------------------------------------------------------------- #
# ASR validation (gate between Whisper and the LLM)
# --------------------------------------------------------------------------- #
NO_SPEECH_THRESHOLD = _env_float("AZMTH_NO_SPEECH_THRESHOLD", 0.5)
MAX_COMPRESSION_RATIO = _env_float("AZMTH_MAX_COMPRESSION_RATIO", 2.4)  # Whisper's own ceiling
MIN_AVG_LOGPROB = _env_float("AZMTH_MIN_AVG_LOGPROB", -1.0)            # Whisper's own floor
MAX_TOKEN_REPEAT_FRAC = _env_float("AZMTH_MAX_TOKEN_REPEAT_FRAC", 0.6)
MIN_TRANSCRIPT_CHARS = _env_int("AZMTH_MIN_TRANSCRIPT_CHARS", 2)

# --------------------------------------------------------------------------- #
# Diagnostics
# --------------------------------------------------------------------------- #
DEBUG = _env("AZMTH_DEBUG", "0") in ("1", "true", "True", "yes")
DEBUG_DIR = "debug_audio"
