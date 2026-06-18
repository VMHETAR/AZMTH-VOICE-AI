AZMTH Voice Assistant

An end-to-end offline conversational AI assistant built with local speech recognition, local language models, and local text-to-speech.

Features
Fully offline operation
Real-time microphone input
Voice Activity Detection (WebRTC VAD)
Speech segmentation
Local speech-to-text using Whisper
Local reasoning using GGUF language models
Local text-to-speech using Piper
No cloud APIs
Privacy-first architecture
Architecture
Microphone
↓
Audio Recorder
↓
WebRTC VAD
↓
Speech Segmenter
↓
Whisper ASR
↓
Local LLM (GGUF)
↓
Piper TTS
↓
Speaker
Current Stack
Speech Recognition
OpenAI Whisper
Local inference
16 kHz audio pipeline
Voice Activity Detection
WebRTC VAD
Speech segmentation
Silence-based utterance detection
Language Model
Qwen2.5 1.5B Instruct
GGUF format
llama.cpp backend
Fully offline inference
Text To Speech
Piper TTS
Lessac English voice
Local waveform generation

AZMTH aims to build a privacy-first conversational AI that runs entirely on edge devices, enabling intelligent voice interaction without reliance on cloud infrastructure.