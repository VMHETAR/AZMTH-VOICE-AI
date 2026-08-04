import sounddevice as sd
import numpy as np

class AudioRecorder:

    def __init__(self, callback_fn, sample_rate=16000, blocksize=320):
        self.callback_fn = callback_fn
        self.blocksize = blocksize
        self.buffer = np.array([], dtype=np.int16)
        # We request blocksize, but hardware drivers might use different sizes
        self.stream = sd.InputStream(
            samplerate=sample_rate,
            channels=1,
            dtype='int16',
            blocksize=blocksize,
            callback=self._audio_callback
        )

    def _audio_callback(self, indata, frames, time, status):
        # Flatten and append new samples
        flat_data = indata.flatten()
        self.buffer = np.append(self.buffer, flat_data)
        
        # Yield exact blocksize chunks (320 samples = 20ms)
        while len(self.buffer) >= self.blocksize:
            chunk = self.buffer[:self.blocksize].copy()
            self.buffer = self.buffer[self.blocksize:]
            
            # Reshape back to (blocksize, 1) for downstream processing
            self.callback_fn(chunk.reshape(-1, 1))

    def start(self):
        self.stream.start()

    def stop(self):
        self.stream.stop()
        self.stream.close()