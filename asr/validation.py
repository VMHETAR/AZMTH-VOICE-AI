"""
Purpose
-------
Quality gate that sits between Whisper and the LLM and rejects obvious ASR
garbage so that corrupted audio never becomes a spoken response.

Rationale
---------
Whisper (especially small, on quiet or noisy audio) hallucinates: repeated
texts like "वह वह वह ...", near-empty "you", or run-on character repetition.
Qwen will dutifully answer whatever it is given, so the damage happens before
the LLM. Whisper already computes the confidence signals we reuse here
(no_speech_prob, avg_logprob, compression_ratio, token ids), so the gate is
cheap and needs no extra models.

Assumptions
-----------
- The caller passes the statistics returned by WhisperASR.transcribe.
- Thresholds come from config.py and can be tuned per deployment.
- A rejection is a soft failure: the utterance is dropped, not fatal.

Tradeoffs
---------
Very conservative (forgiving) thresholds risk letting hallucinations through;
very strict ones can reject legitimate terse replies like "हाँ". Defaults are
set from the measured spread: real speech has no_speech < 0.2, compression
~1.0-1.6 and avg_logprob > -1, while hallucinations sit far outside those.
"""

import re


class ASRValidation:

    def __init__(self, cfg):
        self.cfg = cfg

    def validate(self, text, no_speech_prob=None, avg_logprob=None,
                 compression_ratio=None, tokens=None):
        """Return (accepted: bool, reason: str). Finishes at the first rule."""
        if not text or len(text.strip()) < self.cfg.MIN_TRANSCRIPT_CHARS:
            return False, "empty"

        if no_speech_prob is not None and no_speech_prob > self.cfg.NO_SPEECH_THRESHOLD:
            return False, f"no_speech({no_speech_prob:.2f})"

        if compression_ratio is not None and compression_ratio > self.cfg.MAX_COMPRESSION_RATIO:
            return False, f"garbled(compression={compression_ratio:.2f})"

        if avg_logprob is not None and avg_logprob < self.cfg.MIN_AVG_LOGPROB:
            return False, f"low_confidence(logprob={avg_logprob:.2f})"

        repeat = self._token_repeat_fraction(tokens)
        if repeat > self.cfg.MAX_TOKEN_REPEAT_FRAC:
            return False, f"repetition({repeat:.2f})"

        return True, "ok"

    @staticmethod
    def _token_repeat_fraction(tokens):
        """Fraction of adjacent token pairs that are identical; Whisper's main
        hallucination mode is repeating the same token over and over."""
        if not tokens or len(tokens) < 4:
            return 0.0
        same = sum(1 for i in range(len(tokens) - 1) if tokens[i] == tokens[i + 1])
        return same / (len(tokens) - 1)