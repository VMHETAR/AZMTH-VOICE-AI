import re
import numpy as np
import wave
from piper import PiperVoice

# Latin (?!.) and Devanagari danda (। ॥) sentence boundaries.
SENTENCE_END = re.compile(r"(?<=[.!?।॥])\s+")

def split_sentences(text):
    parts = SENTENCE_END.split(text.strip())
    return [part.strip() for part in parts if part.strip()]


def pop_complete_sentences(buffer):
    sentences = []

    while True:
        match = re.search(r"[.!?।॥](?:\s|$)", buffer)
        if not match:
            break

        end = match.end()
        sentence = buffer[:end].strip()
        if sentence:
            sentences.append(sentence)
        buffer = buffer[end:].lstrip()

    return sentences, buffer


class PiperTTS:

    def __init__(self, model_path):
        self.voice = PiperVoice.load(model_path)
        self.sample_rate = self.voice.config.sample_rate

    def synthesize_chunks(self, text):
        for chunk in self.voice.synthesize(text):
            audio = np.frombuffer(
                chunk.audio_int16_bytes,
                dtype=np.int16,
            ).astype(np.float32) / 32768.0

            yield audio, chunk.sample_rate

    def speak(self, text):
        with wave.open("response.wav", "wb") as wav_file:
            self.voice.synthesize_wav(text, wav_file)
