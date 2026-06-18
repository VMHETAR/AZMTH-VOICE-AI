from llama_cpp import Llama

class LocalLLM:

    def __init__(self):
        self.model = Llama(
            model_path="models/qwen2.5-1.5b-instruct-q4_k_m.gguf",
            n_ctx=2048,
            verbose=False
        )

    def generate(self, prompt):

        formatted_prompt = (
            "<|im_start|>system\n"
            "You are a helpful AI voice assistant.\n"
            "<|im_end|>\n"
            "<|im_start|>user\n"
            f"{prompt}\n"
            "<|im_end|>\n"
            "<|im_start|>assistant\n"
        )

        output = self.model(
            formatted_prompt,
            max_tokens=128,
            temperature=0.7,
            stop=["<|im_end|>"]
        )

        return output["choices"][0]["text"].strip()