import whisper

class WhisperASR:
    def __init__(self, model_name='small'):
        self.model = whisper.load_model(model_name)
    def transribe(self, audio_path):
        result = self.model.transcribe(audio_path)
        return result['text']