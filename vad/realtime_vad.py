import collections
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

    def __init__(self, pre_speech_duration_ms=400, sample_rate=16000, frame_duration_ms=20):
        self.sample_rate = sample_rate
        self.frame_duration_ms = frame_duration_ms
        self.max_pre_speech_frames = int(pre_speech_duration_ms / frame_duration_ms)
        self.pre_speech_buffer = collections.deque(maxlen=self.max_pre_speech_frames)
        
        self.buffer = []
        self.recording = False
        self.silence_frames = 0

        # 400 chunks × 20ms ≈ 8 seconds hard timeout
        self.max_frames = 400

    def reset(self):
        self.pre_speech_buffer.clear()
        self.buffer = []
        self.recording = False
        self.silence_frames = 0

    def add_chunk(self, chunk, is_speech):
        if is_speech:
            self.silence_frames = 0

            if not self.recording:
                self.recording = True
                # Prepend all cached pre-speech frames
                self.buffer.extend(self.pre_speech_buffer)
                self.pre_speech_buffer.clear()

            self.buffer.append(chunk)

            if len(self.buffer) >= self.max_frames:
                segment = np.concatenate(self.buffer, axis=0)
                duration = len(segment) / self.sample_rate

                self.buffer = []
                self.recording = False
                self.silence_frames = 0

                if duration < 0.5:
                    return None

                print("Hard timeout reached")
                return segment

        else:
            if self.recording:
                self.silence_frames += 1
                self.buffer.append(chunk)

                # 30 frames × 20ms = 600ms silence timeout
                if self.silence_frames > 30:
                    segment = np.concatenate(self.buffer, axis=0)
                    duration = len(segment) / self.sample_rate

                    self.buffer = []
                    self.recording = False
                    self.silence_frames = 0

                    if duration < 0.5:
                        return None

                    return segment
            else:
                # Cache pre-speech frames
                self.pre_speech_buffer.append(chunk)

        return None
