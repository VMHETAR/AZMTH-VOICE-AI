import os
import threading
from llama_cpp import Llama

class LocalLLM:

    # Sampling defaults tuned for a spoken voice agent: low temperature and a
    # mild repeat penalty keep short answers stable and stop the 1.5B model from
    # rambling, which matters for Hindi/Devanagari responses in particular.
    GENERATION_KWARGS = dict(
        max_tokens=128,
        temperature=0.6,
        top_p=0.9,
        top_k=40,
        repeat_penalty=1.15,
        stop=["<|im_end|>", "<|im_start|>"],
    )

    # Multilingual voice agent system prompt: stresses language mirroring,
    # concise spoken-style answers, and correct Devanagari danda (।) punctuation
    # so the TTS stream can start as soon as a sentence is complete.
    SYSTEM_PROMPT = (
        "You are a concise voice assistant that speaks out loud. "
        "ALWAYS answer in exactly the same language the user just spoke "
        "(Hindi in Devanagari -> Hindi, English -> English). "
        "Reply with 1-2 very short sentences. End every sentence with "
        "proper punctuation (. ! ? or ।). Never ramble, never list, never repeat. "
        "If you don't know, say so briefly."
    )

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
            {"role": "system", "content": self.SYSTEM_PROMPT}
        ]
        self.lock = threading.Lock()

    def warmup(self):
        """Emit a few tokens with a standalone prompt to absorb one-time load
        costs (tokenizer init, weight page faulting, KV cache allocation) so the
        first real user turn starts from an already-warm model."""
        self.model(
            "<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n",
            max_tokens=8,
            temperature=0.0,
            stop=["<|im_end|>", "<|im_start|>"],
        )

    def reset_history(self):
        with self.lock:
            self.history = [
                {"role": "system", "content": self.SYSTEM_PROMPT}
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

        output = self.model(formatted_prompt, **self.GENERATION_KWARGS)

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
            stream=True,
            **self.GENERATION_KWARGS,
        )

        for output in stream:
            if pipeline_controller and pipeline_id is not None:
                if pipeline_controller.is_cancelled(pipeline_id):
                    break
            
            token = output["choices"][0]["text"]
            if token:
                yield token

