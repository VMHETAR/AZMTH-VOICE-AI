from llama_cpp import Llama


class LocalLLM:

    def __init__(self):
        self.model = Llama(
            model_path="models/qwen2.5-1.5b-instruct-q4_k_m.gguf",
            n_ctx=2048,
            verbose=False,
        )

    def _format_prompt(self, prompt):
        return (
            "<|im_start|>system\n"
            "You are a helpful AI voice assistant.\n"
            "<|im_end|>\n"
            "<|im_start|>user\n"
            f"{prompt}\n"
            "<|im_end|>\n"
            "<|im_start|>assistant\n"
        )

    def generate(self, prompt):
        output = self.model(
            self._format_prompt(prompt),
            max_tokens=128,
            temperature=0.7,
            stop=["<|im_end|>"],
        )

        return output["choices"][0]["text"].strip()

    def generate_stream(self, prompt):
        stream = self.model(
            self._format_prompt(prompt),
            max_tokens=128,
            temperature=0.7,
            stop=["<|im_end|>"],
            stream=True,
        )

        for output in stream:
            token = output["choices"][0]["text"]
            if token:
                yield token
