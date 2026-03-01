import os
from huggingface_hub import InferenceClient

class LLM:
    def __init__(self, model="openai/gpt-oss-20b:groq", api_key=None):
        self.model = model
        self.api_key = api_key or os.environ.get("HF_TOKEN")
        if not self.api_key:
            raise ValueError("HuggingFace API token not found. Please set HF_TOKEN in your environment or pass api_key explicitly.")
        self.client = InferenceClient(api_key=self.api_key)

    def generate(self, prompt, max_tokens=512):
        messages = [
            {"role": "user", "content": prompt}
        ]
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=0.7
        )
        return completion.choices[0].message.content

# Example usage
if __name__ == "__main__":
    llm = LLM()
    answer = llm.generate("What is the capital of France?")
    print("LLM answer:", answer)
