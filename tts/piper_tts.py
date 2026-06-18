from piper import PiperVoice
import wave

class PiperTTS:
    def __init__(self, model_path):
        self.voice = PiperVoice.load(model_path)

    def speak(self, text):
        with wave.open(
            "response.wav",
            "wb"
        ) as wav_file:
            self.voice.synthesize_wav(
                text,
                wav_file
            )
