"""
Purpose: Runs the complete optimized voice AI pipeline.
"""
print("-----------------------------------------------------------------")
print("  Offline Edge AI Voice Assistant is launching...              ")
print("  [Startup] Starting microphone capture & active listening...    ")
print("-----------------------------------------------------------------")

import queue
import threading
import time
import re
import numpy as np

from audio.recorder import AudioRecorder
from audio.audio_output import AudioOutput
from vad.realtime_vad import VoiceActivityDetector, SpeechSegmenter
from pipeline.controller import PipelineController

SAMPLE_RATE = 16000
BARGE_IN_PEAK = 4000
BARGE_IN_FRAMES = 5
MIN_SEGMENT_DURATION = 0.5  # Optimized from 1.0 for short commands
MIN_SEGMENT_PEAK = 3000

# Queues
asr_queue = queue.Queue()
llm_queue = queue.Queue()
tts_queue = queue.Queue()

# Thread-safety locks and active states
active_sentences_lock = threading.Lock()
active_spoken_sentences = []

current_pipeline_id = 0
capture_armed = False
was_assistant_busy = False
interrupt_frames = 0

# Initialize core lightweight components
speaker = AudioOutput()
vad = VoiceActivityDetector()
segmenter = SpeechSegmenter()
pipeline = PipelineController()

# Placeholders and synchronization events for heavy models
asr = None
tts = None
llm = None

asr_ready = threading.Event()
tts_ready = threading.Event()
llm_ready = threading.Event()


def load_asr_background():
    global asr
    try:
        print("[Startup] Loading Speech-to-Text (Whisper)...")
        from asr.whisper_asr import WhisperASR
        asr = WhisperASR("small")
        print("[Startup] Speech-to-Text model loaded successfully.")
    except Exception as e:
        print(f"[Startup] ERROR loading Whisper: {e}")
    finally:
        asr_ready.set()


def load_tts_background():
    global tts
    try:
        print("[Startup] Loading Text-to-Speech (Piper)...")
        from tts.piper_tts import PiperTTS
        tts = PiperTTS("tts/en_US-lessac-medium.onnx")
        print("[Startup] Text-to-Speech model loaded successfully.")
    except Exception as e:
        print(f"[Startup] ERROR loading Piper: {e}")
    finally:
        tts_ready.set()


def load_llm_background():
    global llm
    try:
        print("[Startup] Loading Large Language Model (Qwen)...")
        from llm.local_llm import LocalLLM
        llm = LocalLLM()
        print("[Startup] Large Language Model loaded successfully.")
    except Exception as e:
        print(f"[Startup] ERROR loading LLM: {e}")
    finally:
        llm_ready.set()


def pop_complete_sentences(buffer):
    sentences = []
    while True:
        match = re.search(r"[.!?](?:\s|$)", buffer)
        if not match:
            break
        end = match.end()
        sentence = buffer[:end].strip()
        if sentence:
            sentences.append(sentence)
        buffer = buffer[end:].lstrip()
    return sentences, buffer


def is_assistant_speaking():
    return speaker.is_playing or speaker.has_pending()


def is_assistant_busy():
    return pipeline.is_active() or is_assistant_speaking()


def drain_queue(q):
    while True:
        try:
            q.get_nowait()
        except queue.Empty:
            break


def detect_barge_in(chunk, speech):
    global interrupt_frames

    if not is_assistant_busy():
        interrupt_frames = 0
        return False

    peak = np.max(np.abs(chunk))

    # Dynamic Echo Suppression: raise threshold proportionally to speaker playback amplitude
    speaker_amplitude = speaker.current_amplitude if speaker.is_playing else 0.0
    dynamic_threshold = BARGE_IN_PEAK + 0.6 * speaker_amplitude

    if speech and peak > dynamic_threshold:
        interrupt_frames += 1
    else:
        interrupt_frames = 0

    if interrupt_frames > BARGE_IN_FRAMES:
        interrupt_frames = 0
        return True

    return False


def handle_barge_in():
    global capture_armed, current_pipeline_id

    print("\n>>> BARGE-IN DETECTED <<<")
    
    # Cancel the current transaction
    pipeline.cancel()
    
    # Instantly stop playback
    speaker.stop(current_pipeline_id)
    
    # Drain all queues
    drain_queue(asr_queue)
    drain_queue(llm_queue)
    drain_queue(tts_queue)
    
    # Save the partial assistant response to the LLM history
    with active_sentences_lock:
        if active_spoken_sentences:
            spoken_text = " ".join(active_spoken_sentences)
            if llm is not None:
                llm.add_assistant_message(spoken_text)
            print(f"[LLM History] Added partial assistant response (interrupted): '{spoken_text}'")
            active_spoken_sentences.clear()
            
    segmenter.reset()
    capture_armed = True


def audio_input_callback(indata):
    global capture_armed, was_assistant_busy, current_pipeline_id

    # indata shape is (320, 1), dtype int16
    chunk = indata
    
    try:
        speech = vad.is_speech(chunk)
    except Exception:
        # Ignore malformed chunks
        return

    assistant_busy = is_assistant_busy()

    # Reset segmenter and disarm capture when transitioning from busy to idle
    if was_assistant_busy and not assistant_busy:
        segmenter.reset()
        capture_armed = False

    was_assistant_busy = assistant_busy

    # Run barge-in check
    if detect_barge_in(chunk, speech):
        handle_barge_in()
        return

    # If assistant is speaking and capture is not armed, ignore the audio for speech segmentation
    if is_assistant_speaking() and not capture_armed:
        return

    # Add chunk to segmenter
    segment = segmenter.add_chunk(chunk, speech)

    if segment is None:
        return

    duration = len(segment) / SAMPLE_RATE
    peak = np.max(np.abs(segment))

    print(f"\n[Recorder] Segment captured: duration={duration:.2f}s, peak={peak}")

    if duration < MIN_SEGMENT_DURATION or peak < MIN_SEGMENT_PEAK:
        return

    capture_armed = False

    # Stop active playback if a new query segment starts
    if pipeline.is_active():
        pipeline.cancel()
        speaker.stop(current_pipeline_id)
        
        # Save any spoken sentences before starting new one
        with active_sentences_lock:
            if active_spoken_sentences:
                spoken_text = " ".join(active_spoken_sentences)
                if llm is not None:
                    llm.add_assistant_message(spoken_text)
                active_spoken_sentences.clear()

    # Initialize a new transaction pipeline
    new_id = pipeline.new_pipeline()
    current_pipeline_id = new_id
    
    with active_sentences_lock:
        active_spoken_sentences.clear()

    asr_queue.put((segment, new_id))


# Instantiate recorder with our callback
recorder = AudioRecorder(callback_fn=audio_input_callback)


def asr_worker_loop():
    # Block until Whisper is ready
    asr_ready.wait()
    while True:
        segment, pipeline_id = asr_queue.get()
        if pipeline.is_cancelled(pipeline_id):
            continue

        print(f"[ASR] Transcribing segment for ID {pipeline_id}...")
        try:
            # Transcribe directly from memory
            transcript = asr.transcribe(segment)
            print(f"[ASR] Finished transcription: '{transcript}'")

            if not transcript:
                pipeline.end_pipeline(pipeline_id)
                continue

            if pipeline.is_cancelled(pipeline_id):
                continue

            llm_queue.put((transcript, pipeline_id))
        except Exception as e:
            print(f"[ASR] Error during transcription: {e}")
            pipeline.end_pipeline(pipeline_id)


def llm_worker_loop():
    # Block until LLM is ready
    llm_ready.wait()
    while True:
        transcript, pipeline_id = llm_queue.get()
        if pipeline.is_cancelled(pipeline_id):
            continue

        print(f"[LLM] Generating response for: '{transcript}' (ID {pipeline_id})...")
        try:
            # Streams tokens, checking cancellation along the way
            stream = llm.generate_stream(transcript, pipeline_id, pipeline)
            text_buffer = ""
            sentence_index = 0

            for token in stream:
                if pipeline.is_cancelled(pipeline_id):
                    break

                text_buffer += token
                sentences, text_buffer = pop_complete_sentences(text_buffer)

                for sentence in sentences:
                    if pipeline.is_cancelled(pipeline_id):
                        break
                    tts_queue.put((sentence, sentence_index, pipeline_id))
                    sentence_index += 1

            remainder = text_buffer.strip()
            if remainder and not pipeline.is_cancelled(pipeline_id):
                tts_queue.put((remainder, sentence_index, pipeline_id))
                sentence_index += 1

            # End of response sentinel
            tts_queue.put((None, None, pipeline_id))

        except Exception as e:
            print(f"[LLM] Error during generation: {e}")
            pipeline.end_pipeline(pipeline_id)


def tts_worker_loop():
    # Block until Piper TTS is ready
    tts_ready.wait()
    while True:
        sentence, sentence_index, pipeline_id = tts_queue.get()
        if pipeline.is_cancelled(pipeline_id):
            continue

        # Handle sentinel marking the completion of LLM output
        if sentence is None:
            if not pipeline.is_cancelled(pipeline_id):
                with active_sentences_lock:
                    if active_spoken_sentences:
                        full_response = " ".join(active_spoken_sentences)
                        if llm is not None:
                            llm.add_assistant_message(full_response)
                        print(f"[LLM History] Added assistant response: '{full_response}'")
                        active_spoken_sentences.clear()
                pipeline.end_pipeline(pipeline_id)
            continue

        print(f"[TTS] Synthesizing sentence {sentence_index}: '{sentence}' (ID {pipeline_id})")
        try:
            for audio, sample_rate in tts.synthesize_chunks(sentence):
                if pipeline.is_cancelled(pipeline_id):
                    break
                speaker.enqueue(audio, sample_rate, pipeline_id)

            # Append synthesized sentence once enqueued
            if not pipeline.is_cancelled(pipeline_id):
                with active_sentences_lock:
                    active_spoken_sentences.append(sentence)
        except Exception as e:
            print(f"[TTS] Error during synthesis: {e}")


# Start background model loading threads
threading.Thread(target=load_asr_background, daemon=True).start()
threading.Thread(target=load_tts_background, daemon=True).start()
threading.Thread(target=load_llm_background, daemon=True).start()

# Spawn background worker loops
threading.Thread(target=asr_worker_loop, daemon=True).start()
threading.Thread(target=llm_worker_loop, daemon=True).start()
threading.Thread(target=tts_worker_loop, daemon=True).start()

# Start non-blocking callback recording stream immediately
recorder.start()

print("-----------------------------------------------------------------")
print("  Voice Assistant is ready and listening in the background!     ")
print("  (Models are loading in parallel background threads)           ")
print("-----------------------------------------------------------------")

try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("\nStopping...")
finally:
    recorder.stop()
    speaker.stop()


