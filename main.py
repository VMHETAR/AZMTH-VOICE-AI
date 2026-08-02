"""
Purpose: Runs the complete voice AI pipeline.
"""
import queue
import threading
import time

import numpy as np
import soundfile as sf

from audio.recorder import AudioRecorder
from audio.audio_output import AudioOutput
from vad.realtime_vad import VoiceActivityDetector, SpeechSegmenter
from asr.whisper_asr import WhisperASR
from tts.piper_tts import PiperTTS, pop_complete_sentences
from llm.local_llm import LocalLLM
from pipeline.controller import PipelineController

SAMPLE_RATE = 16000
BARGE_IN_PEAK = 4000
BARGE_IN_FRAMES = 5
MIN_SEGMENT_DURATION = 1.0
MIN_SEGMENT_PEAK = 3000

recorder = AudioRecorder()
speaker = AudioOutput()
vad = VoiceActivityDetector()
segmenter = SpeechSegmenter()
pipeline = PipelineController()

asr = WhisperASR("small")
tts = PiperTTS("tts/en_US-lessac-medium.onnx")
llm = LocalLLM()

speech_queue = queue.Queue()
interrupt_frames = 0
capture_armed = False
was_assistant_busy = False


def is_assistant_speaking():
    return speaker.is_playing or speaker.has_pending()


def is_assistant_busy():
    return pipeline.is_active() or is_assistant_speaking()


def drain_speech_queue():
    while True:
        try:
            speech_queue.get_nowait()
        except queue.Empty:
            break


def queue_speech(segment):
    drain_speech_queue()
    speech_queue.put(segment.copy())


def detect_barge_in(chunk, speech):
    global interrupt_frames

    if not is_assistant_busy():
        interrupt_frames = 0
        return False

    peak = np.max(np.abs(chunk))

    if speech and peak > BARGE_IN_PEAK:
        interrupt_frames += 1
    else:
        interrupt_frames = 0

    if interrupt_frames > BARGE_IN_FRAMES:
        interrupt_frames = 0
        return True

    return False


def handle_barge_in(chunk, speech):
    global capture_armed

    print(f"BARGE-IN DETECTED Peak={np.max(np.abs(chunk))}")
    pipeline.cancel()
    speaker.stop()
    drain_speech_queue()
    segmenter.reset()
    capture_armed = True
    segmenter.add_chunk(chunk, speech)


def speak_text(text):
    for audio, sample_rate in tts.synthesize_chunks(text):
        if pipeline.is_cancelled():
            return
        speaker.enqueue(audio, sample_rate)


def process_response(transcript):
    text_buffer = ""

    for token in llm.generate_stream(transcript):
        if pipeline.is_cancelled():
            return

        text_buffer += token
        sentences, text_buffer = pop_complete_sentences(text_buffer)

        for sentence in sentences:
            if pipeline.is_cancelled():
                return
            speak_text(sentence)

    remainder = text_buffer.strip()
    if remainder and not pipeline.is_cancelled():
        speak_text(remainder)


def worker():
    while True:
        segment = speech_queue.get()
        pipeline.begin()

        try:
            duration = len(segment) / SAMPLE_RATE
            peak = np.max(np.abs(segment))

            if duration < MIN_SEGMENT_DURATION or peak < MIN_SEGMENT_PEAK:
                continue

            filename = f"speech_{int(time.time() * 1000)}.wav"

            sf.write(
                filename,
                segment.squeeze(),
                SAMPLE_RATE,
                subtype="PCM_16",
            )

            if pipeline.is_cancelled():
                continue

            print("Starting Whisper...")
            transcript = asr.transribe(filename)
            print("Whisper Done")
            print(f"User: {transcript}")

            if pipeline.is_cancelled():
                continue

            print("Generating Response...")
            process_response(transcript)

        finally:
            pipeline.end()


print("Starting Voice Assistant...")

recorder.start()
threading.Thread(target=worker, daemon=True).start()

try:
    while True:
        chunk = recorder.read_chunk()
        speech = vad.is_speech(chunk)

        assistant_busy = is_assistant_busy()

        if was_assistant_busy and not assistant_busy:
            segmenter.reset()
            capture_armed = False

        was_assistant_busy = assistant_busy

        if detect_barge_in(chunk, speech):
            handle_barge_in(chunk, speech)
            continue

        if is_assistant_speaking() and not capture_armed:
            continue

        segment = segmenter.add_chunk(chunk, speech)

        if segment is None:
            continue

        duration = len(segment) / SAMPLE_RATE
        peak = np.max(np.abs(segment))

        print(
            f"Duration={duration:.2f}s "
            f"Peak={peak} "
            f"Frames={len(segment)}"
        )

        if duration < MIN_SEGMENT_DURATION or peak < MIN_SEGMENT_PEAK:
            continue

        capture_armed = False

        if pipeline.is_active():
            pipeline.cancel()
            speaker.stop()

        queue_speech(segment)

except KeyboardInterrupt:
    print("\nStopping...")

finally:
    recorder.stop()
