import webrtcvad
import numpy as np


class VoiceActivityDetector:

    def __init__(self, aggressiveness=3, sample_rate=16000):
        self.sample_rate = sample_rate
        self.vad = webrtcvad.Vad(aggressiveness)

    def is_speech(self, chunk):
        return self.vad.is_speech(
            chunk.tobytes(),
            self.sample_rate
        )


class SpeechSegmenter:

    def __init__(self):
        self.buffer = []
        self.recording = False
        self.silence_frames = 0

        # 400 chunks × 20ms ≈ 8 seconds
        self.max_frames = 400

    def add_chunk(self, chunk, is_speech):

        if is_speech:

            self.silence_frames = 0

            if not self.recording:
                self.recording = True

            self.buffer.append(chunk)

            if len(self.buffer) >= self.max_frames:

                segment = np.concatenate(
                    self.buffer,
                    axis=0
                )

                duration = len(segment) / 16000

                self.buffer = []
                self.recording = False
                self.silence_frames = 0

                if duration < 1.0:
                    return None

                print("Hard timeout reached")

                return segment

        else:

            if self.recording:

                self.silence_frames += 1

                self.buffer.append(chunk)

                if self.silence_frames > 50:

                    segment = np.concatenate(
                        self.buffer,
                        axis=0
                    )

                    duration = len(segment) / 16000

                    self.buffer = []
                    self.recording = False
                    self.silence_frames = 0

                    if duration < 1.0:
                        return None

                    return segment

        return None