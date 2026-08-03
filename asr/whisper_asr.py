import whisper
import torch
import numpy as np

class WhisperASR:
    def __init__(self, model_name='small'):
        # Constrain CPU thread count for torch to avoid core contention and cache thrashing.
        # 4 threads is typically the optimal balance for Whisper on CPU.
        if torch.get_num_threads() > 4:
            torch.set_num_threads(4)
        
        # Load the PyTorch model
        self.model = whisper.load_model(model_name)

    def transcribe(self, audio_data, language="en"):
        # If input is a numpy array, make sure it is 16kHz float32 normalized in [-1.0, 1.0]
        if isinstance(audio_data, np.ndarray):
            audio_data = audio_data.squeeze()
            if audio_data.dtype == np.int16:
                audio_data = audio_data.astype(np.float32) / 32768.0
            elif audio_data.dtype != np.float32:
                audio_data = audio_data.astype(np.float32)

        # Greedy decoding (beam_size=1, temperature=0.0) with explicit language is 2-3x faster on CPU
        result = self.model.transcribe(
            audio_data,
            language=language,
            task="transcribe",
            temperature=0.0,
            beam_size=1,
            best_of=1,
            fp16=False
        )
        return result['text'].strip()

    def transribe(self, audio_data):
        # Backward compatibility wrapper for old typo in main.py
        return self.transcribe(audio_data)