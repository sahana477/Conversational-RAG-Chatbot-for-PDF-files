class Observability:
    def __init__(self):
        pass

    def trace(self, user_query, retrieved_chunks, llm_prompt, llm_response):
        pass
if __name__ == "__main__":
    obs = Observability()
    trace = obs.trace("What is RAG?", ["chunk1", "chunk2"], "Prompt text", "LLM response")
    print("Langfuse trace:", trace)
