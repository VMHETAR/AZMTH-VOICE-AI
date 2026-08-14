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

    def __init__(self, pre_speech_duration_ms=400, silence_tail_ms=600,
                 max_segment_duration=8.0, sample_rate=16000, frame_duration_ms=20):
        self.sample_rate = sample_rate
        self.frame_duration_ms = frame_duration_ms
        self.max_pre_speech_frames = int(pre_speech_duration_ms / frame_duration_ms)
        self.pre_speech_buffer = collections.deque(maxlen=self.max_pre_speech_frames)

        # trailing silence before a segment is committed
        self.silence_tail_frames = int(silence_tail_ms / frame_duration_ms)

        # hard cap on utterance length (silence_tail is not required to fire)
        self.max_frames = int(max_segment_duration * 1000 / frame_duration_ms)

        self.buffer = []
        self.recording = False
        self.silence_frames = 0

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

                # silence tail = `silence_tail_frames` frames of non-speech
                if self.silence_frames > self.silence_tail_frames:
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
