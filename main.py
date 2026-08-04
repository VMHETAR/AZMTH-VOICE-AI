"""
Purpose: Runs the complete optimized voice AI pipeline.
"""
print("-----------------------------------------------------------------")
print("  Offline Edge AI Voice Assistant is launching...              ")
print("  [Startup] Importing heavy AI libraries (this takes ~3s)...     ")
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

# Synchronous imports on the main thread to prevent concurrent import lock deadlocks
from asr.whisper_asr import WhisperASR
from tts.piper_tts import PiperTTS
from llm.local_llm import LocalLLM

print("[Startup] Library imports completed. Initializing audio...")

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
tts_en = None
tts_hi = None
llm = None

asr_ready = threading.Event()
tts_ready = threading.Event()
llm_ready = threading.Event()

# No global active_asr_language variable anymore; language is auto-detected per utterance


def load_asr_background():
    global asr
    try:
        print("[Startup] Loading Speech-to-Text (Whisper)...")
        asr = WhisperASR("small")
        print("[Startup] Speech-to-Text model loaded successfully.")
    except Exception as e:
        print(f"[Startup] ERROR loading Whisper: {e}")
    finally:
        asr_ready.set()


def load_tts_background():
    global tts_en, tts_hi
    try:
        print("[Startup] Loading English Text-to-Speech (Piper)...")
        tts_en = PiperTTS("tts/en_US-lessac-medium.onnx")
        print("[Startup] English Text-to-Speech model loaded successfully.")
    except Exception as e:
        print(f"[Startup] ERROR loading English Piper: {e}")

    try:
        print("[Startup] Loading Hindi Text-to-Speech (Piper)...")
        tts_hi = PiperTTS("tts/hi_IN-priyamvada-medium.onnx")
        print("[Startup] Hindi Text-to-Speech model loaded successfully.")
    except Exception as e:
        print(f"[Startup] ERROR loading Hindi Piper: {e}")
    finally:
        tts_ready.set()


def load_llm_background():
    global llm
    try:
        print("[Startup] Loading Large Language Model (Qwen)...")
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

    # We only check for barge-in if the speaker is actively playing audio
    if speaker.active_pipeline_id is None:
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

    barge_id = speaker.active_pipeline_id
    if barge_id is None:
        barge_id = current_pipeline_id

    print(f"\n==================================================")
    print(f"[Barge-In] Barge detected (Target ID: {barge_id})")
    
    # Cancel the target transaction
    pipeline.cancel_id(barge_id)
    print(f"[Barge-In] Pipeline cancelled (ID: {barge_id})")
    
    # Instantly stop playback
    speaker.stop(barge_id)
    print(f"[Barge-In] Speaker stopped")
    
    # Drain all queues
    drain_queue(asr_queue)
    drain_queue(llm_queue)
    drain_queue(tts_queue)
    print(f"[Barge-In] Queue cleared")
    
    # Save the partial assistant response to the LLM history
    with active_sentences_lock:
        if active_spoken_sentences:
            spoken_text = " ".join(active_spoken_sentences)
            if llm is not None:
                llm.add_assistant_message(spoken_text)
            print(f"[LLM History] Added partial assistant response (interrupted): '{spoken_text}'")
            active_spoken_sentences.clear()
            
    segmenter.reset()
    print(f"[Barge-In] Segmenter reset")
    
    capture_armed = True
    print(f"[Barge-In] Capture armed")
    
    # Print diagnostic state
    print(f"[Barge-In] State details:")
    print(f"   - Pipeline active:        {pipeline.is_active()}")
    print(f"   - Pipeline cancelled (ID): {pipeline.is_cancelled(barge_id)}")
    print(f"   - Speaker playing:        {speaker.is_playing}")
    print(f"   - Pending audio:          {speaker.has_pending()}")
    print(f"   - Playback queue size:    {speaker.queue.qsize()}")
    print(f"   - Interrupt frame count:  {interrupt_frames}")
    print(f"   - capture_armed value:    {capture_armed}")
    print(f"==================================================\n")


def audio_input_callback(indata):
    global capture_armed, was_assistant_busy, current_pipeline_id

    # indata shape is (320, 1), dtype int16
    chunk = indata
    
    try:
        speech = vad.is_speech(chunk)
    except Exception as e:
        print(f"[VAD Error] Failed to process chunk: {e}")
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
            # Transcribe directly from memory with auto language detection
            transcript, detected_language = asr.transcribe(segment, language=None, return_language=True)
            
            print(f"--------------------------------------------------")
            print(f"Detected Language: {detected_language}")
            print(f"User Transcript:   '{transcript}'")
            print(f"--------------------------------------------------")

            if not transcript:
                pipeline.end_pipeline(pipeline_id)
                continue

            if pipeline.is_cancelled(pipeline_id):
                continue

            llm_queue.put((transcript, detected_language, pipeline_id))
        except Exception as e:
            print(f"[ASR] Error during transcription: {e}")
            pipeline.end_pipeline(pipeline_id)


def llm_worker_loop():
    # Block until LLM is ready
    llm_ready.wait()
    while True:
        transcript, detected_language, pipeline_id = llm_queue.get()
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
                    tts_queue.put((sentence, sentence_index, detected_language, pipeline_id))
                    sentence_index += 1

            remainder = text_buffer.strip()
            if remainder and not pipeline.is_cancelled(pipeline_id):
                tts_queue.put((remainder, sentence_index, detected_language, pipeline_id))
                sentence_index += 1

            # End of response sentinel
            tts_queue.put((None, None, detected_language, pipeline_id))

        except Exception as e:
            print(f"[LLM] Error during generation: {e}")
            pipeline.end_pipeline(pipeline_id)


def tts_worker_loop():
    # Block until Piper TTS is ready
    tts_ready.wait()
    while True:
        sentence, sentence_index, detected_language, pipeline_id = tts_queue.get()
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

        # Check if the sentence contains Devanagari script (Hindi)
        is_hindi = bool(re.search(r"[\u0900-\u097F]", sentence))
        active_tts = tts_hi if (is_hindi and tts_hi is not None) else tts_en
        
        response_lang = "hi" if is_hindi else "en"
        voice_name = "Hindi (hi_IN-priyamvada-medium)" if is_hindi else "English (en_US-lessac-medium)"

        print(f"--------------------------------------------------")
        print(f"LLM Response Language: {response_lang}")
        print(f"Selected TTS Voice:    {voice_name}")
        print(f"--------------------------------------------------")
        print(f"[TTS] Synthesizing sentence {sentence_index}: '{sentence}' (ID {pipeline_id})")
        try:
            if active_tts is None:
                raise ValueError("Active TTS model is not loaded.")
                
            for audio, sample_rate in active_tts.synthesize_chunks(sentence):
                if pipeline.is_cancelled(pipeline_id):
                    break
                print(f"[TTS Debug] Synthesized audio chunk size={len(audio)}, rate={sample_rate} (ID {pipeline_id})")
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


