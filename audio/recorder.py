import sounddevice as sd

class AudioRecorder:

    def __init__(self):
        self.stream = sd.InputStream(samplerate=16000,channels=1,dtype='int16',blocksize=320)

    def start(self):
        self.stream.start()

    def read_chunk(self):
        chunk, _ = self.stream.read(320)
        return chunk

    def stop(self):
        self.stream.stop()