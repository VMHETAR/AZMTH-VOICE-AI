import os
import threading
from llama_cpp import Llama

class LocalLLM:

    def __init__(self, model_path="models/qwen2.5-1.5b-instruct-q4_k_m.gguf", n_ctx=4096, n_threads=4):
        # Load the model with optimized CPU execution arguments
        # n_threads=4 prevents hyperthreading overhead and CPU core thrashing.
        # use_mlock=True locks the model weights in RAM to prevent disk swapping latency.
        self.model = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_threads=n_threads,
            n_batch=512,
            use_mlock=True,
            use_mmap=True,
            verbose=False,
        )
        self.history = [
            {"role": "system", "content": "You are an offline AI voice assistant. Respond naturally and conversationally. Keep answers concise by default, but always answer the user's question completely. For simple questions, use 1–2 sentences. For explanations, provide enough detail to be accurate while remaining brief. Do not invent facts. If you are unsure, say so."}
        ]
        self.lock = threading.Lock()

    def reset_history(self):
        with self.lock:
            self.history = [
                {"role": "system", "content": "You are an offline AI voice assistant. Respond naturally and conversationally. Keep answers concise by default, but always answer the user's question completely. For simple questions, use 1–2 sentences. For explanations, provide enough detail to be accurate while remaining brief. Do not invent facts. If you are unsure, say so."}
            ]

    def add_user_message(self, text):
        with self.lock:
            self.history.append({"role": "user", "content": text})

    def add_assistant_message(self, text):
        with self.lock:
            self.history.append({"role": "assistant", "content": text})

    def _format_history(self):
        with self.lock:
            prompt = ""
            for msg in self.history:
                prompt += f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>\n"
            prompt += "<|im_start|>assistant\n"
            return prompt

    def generate(self, prompt):
        # Stateful helper for non-streaming completions
        self.add_user_message(prompt)
        formatted_prompt = self._format_history()
        
        output = self.model(
            formatted_prompt,
            max_tokens=128,
            temperature=0.7,
            stop=["<|im_end|>", "<|im_start|>"],
        )
        
        response_text = output["choices"][0]["text"].strip()
        self.add_assistant_message(response_text)
        return response_text

    def generate_stream(self, prompt, pipeline_id=None, pipeline_controller=None):
        # We format prompt using the current history (prompt is already user's transcript).
        # We must add user message to history first.
        self.add_user_message(prompt)
        formatted_prompt = self._format_history()

        stream = self.model(
            formatted_prompt,
            max_tokens=128,
            temperature=0.7,
            stop=["<|im_end|>", "<|im_start|>"],
            stream=True,
        )

        for output in stream:
            if pipeline_controller and pipeline_id is not None:
                if pipeline_controller.is_cancelled(pipeline_id):
                    break
            
            token = output["choices"][0]["text"]
            if token:
                yield token

