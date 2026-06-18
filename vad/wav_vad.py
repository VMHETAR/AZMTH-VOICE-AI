import collections
import contextlib
import sys
import wave

def write_path(path, audio, sample_rate):
    with contextlib.closing(wave.open(path, "wb")) as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio)


class Frame:
    def __init__(self, bytes_data, timestamp, duration):
        self.bytes = bytes_data
        self.timestamp = timestamp
        self.duration = duration


def frame_generator(frame_duration_ms, audio, sample_rate):
    bytes_per_frame = int(
        sample_rate * (frame_duration_ms / 1000.0) * 2
    )

    offset = 0
    timestamp = 0.0
    duration = (bytes_per_frame / sample_rate) / 2.0

    while offset + bytes_per_frame <= len(audio):
        yield Frame(
            audio[offset: offset + bytes_per_frame],
            timestamp,
            duration,
        )

        timestamp += duration
        offset += bytes_per_frame


def vad_collector(sample_rate,frame_duration_ms,padding_duration_ms,vad,frames):
    num_padding_frames = int(
        padding_duration_ms / frame_duration_ms
    )

    ring_buffer = collections.deque(
        maxlen=num_padding_frames
    )

    triggered = False
    voiced_frames = []

    for frame in frames:

        is_speech = vad.is_speech(frame.bytes,sample_rate)

        sys.stdout.write("1" if is_speech else "0")

        if not triggered:

            ring_buffer.append(
                (frame, is_speech)
            )

            num_voiced = sum(
                speech
                for _, speech in ring_buffer
            )

            if num_voiced > 0.9 * ring_buffer.maxlen:

                triggered = True

                sys.stdout.write(
                    f"+({ring_buffer[0][0].timestamp})"
                )

                for buffered_frame, _ in ring_buffer:
                    voiced_frames.append(
                        buffered_frame
                    )

                ring_buffer.clear()

        else:

            voiced_frames.append(frame)

            ring_buffer.append(
                (frame, is_speech)
            )

            num_unvoiced = sum(
                not speech
                for _, speech in ring_buffer
            )

            if num_unvoiced > 0.9 * ring_buffer.maxlen:

                sys.stdout.write(
                    f"-({frame.timestamp + frame.duration})"
                )

                yield b"".join(
                    f.bytes
                    for f in voiced_frames
                )

                triggered = False
                ring_buffer.clear()
                voiced_frames = []

    if triggered:
        sys.stdout.write(
            f"-({frame.timestamp + frame.duration})"
        )

    sys.stdout.write("\n")

    if voiced_frames:
        yield b"".join(
            f.bytes
            for f in voiced_frames
        )