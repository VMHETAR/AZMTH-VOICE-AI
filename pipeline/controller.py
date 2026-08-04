import threading

class PipelineController:

    def __init__(self):
        self._lock = threading.Lock()
        self._current_id = 0
        self._cancelled_ids = set()
        self._active = False

    def new_pipeline(self):
        with self._lock:
            self._current_id += 1
            self._active = True
            return self._current_id

    def cancel(self):
        with self._lock:
            self._cancelled_ids.add(self._current_id)
            self._active = False

    def cancel_id(self, pipeline_id):
        with self._lock:
            self._cancelled_ids.add(pipeline_id)
            if pipeline_id == self._current_id:
                self._active = False

    def is_cancelled(self, pipeline_id):
        with self._lock:
            return pipeline_id in self._cancelled_ids

    def is_active(self):
        with self._lock:
            return self._active

    def end_pipeline(self, pipeline_id):
        with self._lock:
            if pipeline_id == self._current_id:
                self._active = False
            # Clean up old cancelled ids to keep memory footprint bounded
            self._cancelled_ids.discard(pipeline_id - 5)
