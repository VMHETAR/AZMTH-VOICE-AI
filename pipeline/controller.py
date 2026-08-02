import threading

class PipelineController:

    def __init__(self):
        self._lock = threading.Lock()
        self._cancel = threading.Event()
        self._active = False

    def begin(self):
        with self._lock:
            self._cancel.clear()
            self._active = True

    def end(self):
        with self._lock:
            self._active = False

    def cancel(self):
        self._cancel.set()

    def is_cancelled(self):
        return self._cancel.is_set()

    def is_active(self):
        with self._lock:
            return self._active
