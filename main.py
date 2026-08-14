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
import os
import wave
import numpy as np

import config

from audio.recorder import AudioRecorder
from audio.audio_output import AudioOutput
from vad.realtime_vad import VoiceActivityDetector, SpeechSegmenter
from pipeline.controller import PipelineController
from pipeline.metrics import metrics

# Synchronous imports on the main thread to prevent concurrent import lock deadlocks
from asr.whisper_asr import WhisperASR
from asr.validation import ASRValidation
from tts.piper_tts import PiperTTS
from llm.local_llm import LocalLLM

print("[Startup] Library imports completed. Initializing audio...")

SAMPLE_RATE = config.SAMPLE_RATE
BARGE_IN_PEAK = config.BARGE_IN_PEAK
BARGE_IN_FRAMES = config.BARGE_IN_FRAMES
MIN_SEGMENT_DURATION = config.MIN_SEGMENT_DURATION
MIN_SEGMENT_PEAK = config.MIN_SEGMENT_PEAK

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
last_language = None  # stable ASR language across utterances (ASR worker only)

# Initialize core lightweight components
speaker = AudioOutput()
vad = VoiceActivityDetector(aggressiveness=config.VAD_AGGRESSIVENESS)
segmenter = SpeechSegmenter(pre_speech_duration_ms=config.PRE_SPEECH_MS,
                            silence_tail_ms=config.SILENCE_TAIL_MS,
                            max_segment_duration=config.MAX_SEGMENT_DURATION)
asr_validator = ASRValidation(config)
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
        asr = WhisperASR()
        print(f"[Startup] Speech-to-Text model '{asr.model_name}' loaded successfully.")
        asr.warmup()
        print("[Startup] Whisper warm-up complete (first transcription skips cold-start).")
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
        llm.warmup()
        print("[Startup] Large Language Model loaded successfully.")
    except Exception as e:
        print(f"[Startup] ERROR loading LLM: {e}")
    finally:
        llm_ready.set()


def pop_complete_sentences(buffer):
    sentences = []
    while True:
        # Split on Latin (?!.), and Devanagari danda (। ॥) punctuation so that
        # Hindi/Devanagari responses stream to TTS sentence-by-sentence too.
        match = re.search(r"[.!?।॥](?:\s|$)", buffer)
        if not match:
            break
        end = match.end()
        sentence = buffer[:end].strip()
        if sentence:
            sentences.append(sentence)
        buffer = buffer[end:].lstrip()
    return sentences, buffer


def select_tts_for_language(detected_language, sentence):
    """Choose the TTS voice from the language Whisper detected, falling back
    to script-based detection when the language is unsupported/unknown."""
    voice_map = {"en": tts_en, "hi": tts_hi}

    # 1) Prefer the detected language mapping.
    if detected_language in voice_map:
        return voice_map[detected_language], detected_language

    # 2) Fallback: detect script when Whisper language code is unknown.
    if re.search(r"[\u0900-\u097F]", sentence):
        return tts_hi, "hi"
    return tts_en, "en"


def resolve_language(detected, confidence):
    """Keep language detection stable across utterances.

    Whisper's language head is near-random on quiet/noisy clips (top-1
    confidence ~0.25-0.35), so re-detecting per utterance causes flicker.
    Trust a detection only when confident; otherwise reuse the last confirmed
    language. config.DEFAULT_LANG (AZMTH_LANG) always wins when set.
    """
    global last_language

    if config.DEFAULT_LANG:
        return config.DEFAULT_LANG, True
    if confidence >= config.LANG_CONF_THRESHOLD:
        last_language = detected
        return detected, True
    if last_language is not None:
        return last_language, False
    return detected, False


def save_segment_wav(segment, path):
    """Write an int16 segment to disk for offline inspection (DEBUG mode only)."""
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(np.asarray(segment, dtype="<i2").tobytes())


def log_asr_diagnostics(pipeline_id, duration, peak, detected, confidence,
                        stats, accepted, reason):
    """One structured diagnostic block per utterance, only when AZMTH_DEBUG=1."""
    if not config.DEBUG:
        return
    print("\n==================== ASR DIAGNOSTIC ====================")
    print(f"[Audio] sample_rate={SAMPLE_RATE} channels=1 dtype=int16 "
          f"duration={duration:.2f}s peak={peak} rms={stats.get('rms', 0):.0f}")
    print(f"[VAD]   pre_pad_ms={config.PRE_SPEECH_MS} tail_ms={config.SILENCE_TAIL_MS} "
          f"min_seg_s={MIN_SEGMENT_DURATION} max_seg_s={config.MAX_SEGMENT_DURATION}")
    print(f"[ASR]   model={asr.model_name} backend=pytorch-cpu "
          f"lang={detected.lower() if detected else '?'} lang_conf={confidence:.3f} "
          f"transcript='{stats.get('text', '')}'")
    print(f"[ASR]   no_speech_prob={stats.get('no_speech_prob')} "
          f"avg_logprob={stats.get('avg_logprob')} compression_ratio={stats.get('compression_ratio')} "
          f"tokens={len(stats.get('tokens') or [])}")
    print(f"[Validation] accepted={accepted} reason={reason}")
    print(f"[LLM]   skipped={'YES (rejected)' if not accepted else 'no'}")
    print(f"[End-to-End] pipeline_id={pipeline_id} asr_time={stats.get('asr_time', 0):.3f}s")
    print("=========================================================\n")


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

    metrics.mark_barge_in(barge_id)

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
        metrics.count("segments_rejected_short")
        return

    metrics.count("segments_captured")
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

    # eos_time = the moment speech ends; every downstream latency is measured from here
    metrics.start_pipeline(new_id, time.perf_counter())
    metrics.count("utterances")

    if config.DEBUG:
        os.makedirs(config.DEBUG_DIR, exist_ok=True)
        save_segment_wav(segment, os.path.join(config.DEBUG_DIR, f"segment_{new_id:04d}.wav"))

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

        segment = np.asarray(segment).reshape(-1)
        segment_duration = len(segment) / SAMPLE_RATE
        segment_peak = int(np.max(np.abs(segment)))

        print(f"[ASR] Transcribing segment {segment_duration:.2f}s for ID {pipeline_id}...")
        t_asr_start = time.perf_counter()
        try:
            # 1) Language detection with confidence (near-random on garbage audio)
            detected, confidence = asr.detect_language(segment)

            # 2) Stable across utterances; explicit language also skips the
            #    redundant internal detection pass inside Whisper.
            resolved_lang, trusted = resolve_language(detected, confidence)

            # 3) Transcribe with the resolved language + per-utterance stats
            transcript, out_lang, stats = asr.transcribe(
                segment, language=resolved_lang, return_stats=True)
            t_asr_end = time.perf_counter()
            stats["asr_time"] = t_asr_end - t_asr_start
            stats["text"] = transcript
            stats["rms"] = float(np.sqrt(np.mean(segment.astype(np.float32) ** 2)))

            metrics.record_asr(pipeline_id, t_asr_start, t_asr_end,
                               out_lang, len(transcript))

            # 4) ASR validation gate: NEVER let hallucinations reach the LLM
            accepted, reason = asr_validator.validate(
                transcript,
                no_speech_prob=stats["no_speech_prob"],
                avg_logprob=stats["avg_logprob"],
                compression_ratio=stats["compression_ratio"],
                tokens=stats["tokens"],
            )

            log_asr_diagnostics(pipeline_id, segment_duration, segment_peak,
                                detected, confidence, stats, accepted, reason)

            print(f"--------------------------------------------------")
            print(f"Language: {out_lang} (conf={confidence:.2f}{'' if trusted else ', carried over'})")
            print(f"User Transcript:   '{transcript}'")
            print(f"[ASR] Transcription took {t_asr_end - t_asr_start:.3f}s")
            print(f"[ASR] Validation: {'ACCEPTED' if accepted else 'REJECTED'} ({reason})")
            print(f"--------------------------------------------------")

            if not transcript:
                pipeline.end_pipeline(pipeline_id)
                metrics.end_pipeline(pipeline_id)
                continue

            if not accepted:
                metrics.count("asr_rejected")
                pipeline.end_pipeline(pipeline_id)
                metrics.end_pipeline(pipeline_id)
                continue

            if pipeline.is_cancelled(pipeline_id):
                continue

            metrics.count("asr_accepted")
            llm_queue.put((transcript, out_lang, pipeline_id))
        except Exception as e:
            print(f"[ASR] Error during transcription: {e}")
            metrics.record_error("asr")
            pipeline.end_pipeline(pipeline_id)
            metrics.end_pipeline(pipeline_id)


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
            t_first_token = None
            token_count = 0

            for token in stream:
                if pipeline.is_cancelled(pipeline_id):
                    break

                if t_first_token is None:
                    t_first_token = time.perf_counter()
                token_count += 1
                text_buffer += token
                sentences, text_buffer = pop_complete_sentences(text_buffer)

                for sentence in sentences:
                    if pipeline.is_cancelled(pipeline_id):
                        break
                    tts_queue.put((sentence, sentence_index, detected_language, pipeline_id))
                    sentence_index += 1

            if t_first_token is not None:
                t_llm_end = time.perf_counter()
                metrics.record_llm(pipeline_id, t_first_token, t_llm_end,
                                   token_count, len(text_buffer) + token_count)

            remainder = text_buffer.strip()
            if remainder and not pipeline.is_cancelled(pipeline_id):
                tts_queue.put((remainder, sentence_index, detected_language, pipeline_id))
                sentence_index += 1

            # End of response sentinel
            tts_queue.put((None, None, detected_language, pipeline_id))

        except Exception as e:
            print(f"[LLM] Error during generation: {e}")
            metrics.record_error("llm")
            pipeline.end_pipeline(pipeline_id)
            metrics.end_pipeline(pipeline_id)


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
                metrics.end_pipeline(pipeline_id)
            continue

        # Prefer the voice matching Whisper's detected language; fall back to script.
        active_tts, response_lang = select_tts_for_language(detected_language, sentence)
        voice_name = "Hindi (hi_IN-priyamvada-medium)" if response_lang == "hi" else "English (en_US-lessac-medium)"

        print(f"--------------------------------------------------")
        print(f"LLM Response Language: {response_lang}")
        print(f"Selected TTS Voice:    {voice_name}")
        print(f"--------------------------------------------------")
        print(f"[TTS] Synthesizing sentence {sentence_index}: '{sentence}' (ID {pipeline_id})")
        try:
            if active_tts is None:
                raise ValueError("Active TTS model is not loaded.")
                
            t_synth_start = time.perf_counter()
            total_samples = 0
            for audio, sample_rate in active_tts.synthesize_chunks(sentence):
                if pipeline.is_cancelled(pipeline_id):
                    break
                total_samples += len(audio)
                speaker.enqueue(audio, sample_rate, pipeline_id)
            t_synth_end = time.perf_counter()
            metrics.record_tts(pipeline_id, t_synth_start, t_synth_end, total_samples)

            # Append synthesized sentence once enqueued
            if not pipeline.is_cancelled(pipeline_id):
                with active_sentences_lock:
                    active_spoken_sentences.append(sentence)
        except Exception as e:
            print(f"[TTS] Error during synthesis: {e}")
            metrics.record_error("tts")


# Start background model loading threads
threading.Thread(target=load_asr_background, daemon=True).start()
threading.Thread(target=load_tts_background, daemon=True).start()
threading.Thread(target=load_llm_background, daemon=True).start()

# Spawn background worker loops
threading.Thread(target=asr_worker_loop, daemon=True).start()
threading.Thread(target=llm_worker_loop, daemon=True).start()
threading.Thread(target=tts_worker_loop, daemon=True).start()

# Report the instant audio for each pipeline actually hits the sound card,
# which anchors the end-to-end response latency measurement.
speaker.on_first_audio = metrics.record_first_audio


def queue_sampler_loop():
    while True:
        metrics.record_queue_sample(asr_queue.qsize(), llm_queue.qsize(), tts_queue.qsize())
        time.sleep(1)


threading.Thread(target=queue_sampler_loop, daemon=True).start()

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

    # Persist and print latency metrics gathered during this session
    try:
        metrics.to_csv("metrics.csv")
        metrics.to_json("metrics.json")
    except Exception as e:
        print(f"[Metrics] Failed to write reports: {e}")
    metrics.print_summary()


