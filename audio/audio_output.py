import queue
import threading

import numpy as np
import sounddevice as sd


class AudioOutput:

    def __init__(self):
        self.is_playing = False
        self._queue = queue.Queue()
        self._stop_event = threading.Event()
        self._player_thread = threading.Thread(
            target=self._player_loop,
            daemon=True,
        )
        self._player_thread.start()

    def enqueue(self, audio, sample_rate):
        self._stop_event.clear()
        self._queue.put((audio, sample_rate))

    def has_pending(self):
        return not self._queue.empty()

    def _player_loop(self):
        while True:
            try:
                audio, sample_rate = self._queue.get(timeout=0.05)
            except queue.Empty:
                if not self._stop_event.is_set():
                    self.is_playing = False
                continue

            if self._stop_event.is_set():
                continue

            self.is_playing = True
            sd.play(audio, sample_rate)
            sd.wait()

            if self._stop_event.is_set():
                self.is_playing = False
            else:
                self.is_playing = not self._queue.empty()

    def stop(self):
        self._stop_event.set()
        sd.stop()

        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break

        self.is_playing = False
