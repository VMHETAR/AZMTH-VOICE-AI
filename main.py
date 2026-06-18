from audio.recorder import AudioRecorder
from vad.realtime_vad import (
    VoiceActivityDetector,
    SpeechSegmenter
)
from asr.whisper_asr import WhisperASR
from tts.piper_tts import PiperTTS

import numpy as np
import soundfile as sf
import time

from llm.local_llm import LocalLLM
recorder = AudioRecorder()
vad = VoiceActivityDetector()
segmenter = SpeechSegmenter()

asr = WhisperASR("small")

tts = PiperTTS(
    "tts/en_US-lessac-medium.onnx"
)

llm = LocalLLM()
print("Starting Voice Assistant...")

recorder.start()

try:

    while True:

        chunk = recorder.read_chunk()

        speech = vad.is_speech(chunk)

        segment = segmenter.add_chunk(
            chunk,
            speech
        )

        if segment is not None:

            duration = len(segment) / 16000
            peak = np.max(np.abs(segment))

            print(
                f"Duration={duration:.2f}s "
                f"Peak={peak}"
                f"Frames={len(segment)}"
            )

            if duration > 1.0 and peak > 3000:

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
                transcript = asr.transribe(filename)
                print("Whisper Done")

                print(f"User: {transcript}")

                response = f"You said {transcript}"

                print("Starting TTS...")
                tts.speak(response)
                print("TTS Done")

                transcript = asr.transribe(
                    filename
                )

                print(
                    f"User: {transcript}"
                )

                response = llm.generate(
                    transcript
                )
                response = llm.generate(transcript)

                print(type(response))
                print(response)
                print(
                    f"Assistant: {response}"
                )

                tts.speak(
                    response
                )

except KeyboardInterrupt:

    print("\nStopping...")

finally:

    recorder.stop()