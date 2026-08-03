import sounddevice as sd

class AudioRecorder:

    def __init__(self, callback_fn, sample_rate=16000, blocksize=320):
        self.callback_fn = callback_fn
        self.stream = sd.InputStream(
            samplerate=sample_rate,
            channels=1,
            dtype='int16',
            blocksize=blocksize,
            callback=self._audio_callback
        )

    def _audio_callback(self, indata, frames, time, status):
        # We must copy indata because sounddevice reuses the buffer!
        self.callback_fn(indata.copy())

    def start(self):
        self.stream.start()

    def stop(self):
        self.stream.stop()
        self.stream.close()