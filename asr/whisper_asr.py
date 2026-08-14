"""
Purpose
-------
Wraps OpenAI Whisper for the voice pipeline: handles audio dtype/normalization,
applies configurable pre-gain for quiet microphones, exposes a language-detection
primitive with confidence, and returns per-utterance confidence statistics so the
ASR validation gate can reject hallucinations.

Rationale
---------
The pipeline needs more than a transcript: it needs the language confidence to
keep language detection stable across utterances, and the per-utterance stats
(no_speech_prob, avg_logprob, compression_ratio, tokens) that the validation
layer uses to discard garbage before it reaches the LLM.
"""

import time

import numpy as np
import torch
import whisper

import config


class WhisperASR:

    def __init__(self, model_name=None, threads=None, gain=None):
        self.model_name = model_name or config.ASR_MODEL
        self.gain = gain if gain is not None else config.ASR_GAIN

        # 8 threads measured ~1.5x faster than the previous 4 on the reference
        # 18-core CPU (4.2s vs 6.4s on a 2s utterance); 14 threads were slower
        # due to cache thrashing.
        if threads is None:
            threads = config.ASR_THREADS
        torch.set_num_threads(threads)

        self.model = whisper.load_model(self.model_name)

    def _prepare(self, audio_data):
        """Convert raw capture to 16kHz float32 in [-1, 1], applying pre-gain."""
        arr = np.asarray(audio_data).squeeze()
        if arr.dtype == np.int16:
            arr = arr.astype(np.float32) / 32768.0
        elif arr.dtype != np.float32:
            arr = arr.astype(np.float32)
        arr = arr.flatten()
        if self.gain != 1.0:
            arr = arr * self.gain
            np.clip(arr, -1.0, 1.0, out=arr)
        return arr

    def _mel(self, audio):
        audio = np.pad(audio, (0, 16000 * 30 - len(audio)))
        return whisper.log_mel_spectrogram(audio).to(self.model.device)

    def detect_language(self, audio_data):
        """Return (language_code, top1_confidence) via Whisper's language head.

        Whisper's top-1 confidence is near-random (~0.25-0.35) on silence and
        noise; callers should use config.LANG_CONF_THRESHOLD to decide whether
        to trust it."""
        audio = self._prepare(audio_data)
        mel = self._mel(audio)
        _, probs = self.model.detect_language(mel)
        best = max(probs.items(), key=lambda kv: kv[1])
        return best[0], float(best[1])

    def transcribe(self, audio_data, language=None, return_language=False, return_stats=False):
        audio = self._prepare(audio_data)

        # language=None lets Whisper auto-detect; passing an explicit language
        # skips the internal detection pass (used for language stability).
        result = self.model.transcribe(
            audio,
            language=language,
            task="transcribe",
            temperature=0.0,
            beam_size=1,
            best_of=1,
            fp16=False,
            without_timestamps=True,
            condition_on_previous_text=False,
        )
        text = result["text"].strip()
        detected_lang = result.get("language")

        seg = (result.get("segments") or [None])[0]
        stats = {
            "language": detected_lang,
            "no_speech_prob": seg.get("no_speech_prob") if seg is not None else None,
            "avg_logprob": seg.get("avg_logprob") if seg is not None else None,
            "compression_ratio": seg.get("compression_ratio") if seg is not None else None,
            "tokens": seg.get("tokens") if seg is not None else None,
        }

        if return_stats:
            return text, detected_lang, stats
        if return_language:
            return text, detected_lang
        return text

    def warmup(self):
        """Run a tiny no-op transcription so the first real utterance is not
        penalized by one-time model warmup costs."""
        silence = np.zeros(4800, dtype=np.float32)
        self.transcribe(silence)

    def transribe(self, audio_data):
        # Backward compatibility wrapper for old typo in main.py
        return self.transcribe(audio_data)