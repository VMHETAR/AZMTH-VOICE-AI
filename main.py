from audio.recorder import AudioRecorder
from audio.audio_output import AudioOutput

from vad.realtime_vad import (
    VoiceActivityDetector,
    SpeechSegmenter
)

from asr.whisper_asr import WhisperASR
from tts.piper_tts import PiperTTS
from llm.local_llm import LocalLLM

import numpy as np
import soundfile as sf
import time


recorder = AudioRecorder()
speaker = AudioOutput()

vad = VoiceActivityDetector()
segmenter = SpeechSegmenter()

asr = WhisperASR("small")

tts = PiperTTS(
    "tts/en_US-lessac-medium.onnx"
)

llm = LocalLLM()

interrupt_frames = 0

print("Starting Voice Assistant...")

recorder.start()

try:

    while True:

        chunk = recorder.read_chunk()

        speech = vad.is_speech(chunk)

        # --------------------
        # BARGE-IN
        # --------------------

        if speaker.is_playing:

            peak = np.max(np.abs(chunk))

            if speech and peak > 4000:
                interrupt_frames += 1
            else:
                interrupt_frames = 0

            if interrupt_frames > 8:

                print(
                    f"BARGE-IN DETECTED "
                    f"Peak={peak}"
                )

                speaker.stop()

                interrupt_frames = 0

                # IMPORTANT:
                # Do NOT reset segmenter
                # Do NOT continue
                # Let the user's speech continue
                # into the normal pipeline below

        else:

            interrupt_frames = 0

        # --------------------
        # NORMAL PIPELINE
        # --------------------

        segment = segmenter.add_chunk(
            chunk,
            speech
        )

        if segment is None:
            continue

        duration = len(segment) / 16000
        peak = np.max(np.abs(segment))

        print(
            f"Duration={duration:.2f}s "
            f"Peak={peak} "
            f"Frames={len(segment)}"
        )

        if duration < 1.0 or peak < 3000:
            continue

        filename = (
            f"speech_{int(time.time() * 1000)}.wav"
        )

        print("Saving WAV...")

        sf.write(
            filename,
            segment.squeeze(),
            16000,
            subtype="PCM_16"
        )

        print("WAV Saved")

        print("Starting Whisper...")

        transcript = asr.transribe(
            filename
        )

        print("Whisper Done")

        print(
            f"User: {transcript}"
        )

        print(
            "Generating Response..."
        )

        response = llm.generate(
            transcript
        )

        print(
            f"Assistant: {response}"
        )

        print("Generating TTS...")

        tts.speak(
            response
        )

        print("Playing Audio...")

        speaker.play(
            "response.wav"
        )

except KeyboardInterrupt:

    print("\nStopping...")

finally:

    recorder.stop()