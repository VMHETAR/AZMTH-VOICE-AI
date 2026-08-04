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
        self.cancelled_pipeline_ids = set()
        self.active_pipeline_id = None

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
                        self.buffer, self.active_pipeline_id = self.queue.get_nowait()
                    except queue.Empty:
                        self.active_pipeline_id = None
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
                self.active_pipeline_id = None

    def enqueue(self, audio, sample_rate, pipeline_id=None):
        with self.lock:
            if pipeline_id is not None and pipeline_id in self.cancelled_pipeline_ids:
                return
            
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
            
            self.queue.put((audio_flat, pipeline_id))
            self.is_playing = True

    def has_pending(self):
        with self.lock:
            return not self.queue.empty() or len(self.buffer) > 0

    def stop(self, pipeline_id=None):
        with self.lock:
            if pipeline_id is not None:
                self.cancelled_pipeline_ids.add(pipeline_id)
                # Bound size of cancelled set
                if len(self.cancelled_pipeline_ids) > 100:
                    self.cancelled_pipeline_ids = set(list(self.cancelled_pipeline_ids)[-50:])

            # Clear queue
            while not self.queue.empty():
                try:
                    self.queue.get_nowait()
                except queue.Empty:
                    break
            self.buffer = np.array([], dtype=np.float32)
            self._current_amplitude = 0.0
            self.is_playing = False
            self.active_pipeline_id = None
            # Deleting the stream stop/start here avoids PortAudio deadlocking during active callback.
            # Draining queue and buffer will instantly cause the callback to output silence.


