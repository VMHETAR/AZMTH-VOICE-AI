import sounddevice as sd
import soundfile as sf
import threading

class AudioOutput:

    def __init__(self):
        self.is_playing = False

    def play(self, filename):

        thread = threading.Thread(
            target=self._play,
            args=(filename,)
        )

        thread.daemon = True
        thread.start()
    def _play(self, filename):

        audio, sr = sf.read(filename)

        self.is_playing = True

        sd.play(audio, sr)
        sd.wait()

        self.is_playing = False

    def stop(self):

        sd.stop()
        self.is_playing = False