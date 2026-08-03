import queue
import threading
import numpy as np
import sounddevice as sd

class AudioOutput:

    def __init__(self):
        self.stream = None
        self.sample_rate = None
        self.queue = queue.Queue()
        self.buffer = np.array([], dtype=np.float32)
        self.is_playing = False
        self.lock = threading.Lock()
        self._current_amplitude = 0.0

    @property
    def current_amplitude(self):
        return self._current_amplitude

    def _callback(self, outdata, frames, time, status):
        with self.lock:
            needed = frames
            out = np.zeros(frames, dtype=np.float32)
            out_offset = 0

            while needed > 0:
                if len(self.buffer) == 0:
                    try:
                        self.buffer = self.queue.get_nowait()
                    except queue.Empty:
                        break

                n = min(len(self.buffer), needed)
                out[out_offset:out_offset+n] = self.buffer[:n]
                self.buffer = self.buffer[n:]
                out_offset += n
                needed -= n

            outdata[:, 0] = out
            if out_offset > 0:
                self._current_amplitude = float(np.max(np.abs(out)))
                self.is_playing = True
            else:
                self._current_amplitude = 0.0
                self.is_playing = not self.queue.empty()

    def enqueue(self, audio, sample_rate):
        with self.lock:
            # Ensure audio is float32 and flat
            audio_flat = np.asarray(audio, dtype=np.float32).flatten()
            
            if self.stream is None or self.sample_rate != sample_rate:
                if self.stream is not None:
                    try:
                        self.stream.stop()
                        self.stream.close()
                    except Exception:
                        pass
                self.sample_rate = sample_rate
                self.stream = sd.OutputStream(
                    samplerate=sample_rate,
                    channels=1,
                    dtype='float32',
                    blocksize=512,
                    callback=self._callback
                )
                self.stream.start()
            
            self.queue.put(audio_flat)
            self.is_playing = True

    def has_pending(self):
        with self.lock:
            return not self.queue.empty() or len(self.buffer) > 0

    def stop(self):
        with self.lock:
            # Clear queue
            while not self.queue.empty():
                try:
                    self.queue.get_nowait()
                except queue.Empty:
                    break
            self.buffer = np.array([], dtype=np.float32)
            self._current_amplitude = 0.0
            self.is_playing = False
            if self.stream is not None:
                try:
                    # Clear sounddevice buffer
                    self.stream.stop()
                    self.stream.start()
                except Exception:
                    pass

