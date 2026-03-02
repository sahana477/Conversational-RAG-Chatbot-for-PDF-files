import os
from dotenv import load_dotenv
load_dotenv(override=True)
from langfuse import Langfuse
import uuid

class LangfuseTraceBody:
    def __init__(self, input, output, query=None, retrieved_chunks=None, prompt=None, response=None, id=None):
        self.id = id or str(uuid.uuid4())
        self.input = input
        self.output = output
        self.query = query
        self.retrieved_chunks = retrieved_chunks
        self.prompt = prompt
        self.response = response

    def copy(self, update=None):
        update = update or {}
        return LangfuseTraceBody(
            input=update.get("input", self.input),
            output=update.get("output", self.output),
            query=update.get("query", self.query),
            retrieved_chunks=update.get("retrieved_chunks", self.retrieved_chunks),
            prompt=update.get("prompt", self.prompt),
            response=update.get("response", self.response),
            id=update.get("id", self.id)
        )

    def dict(self):
        return {
            "id": self.id,
            "input": self.input,
            "output": self.output,
            "query": self.query,
            "retrieved_chunks": self.retrieved_chunks,
            "prompt": self.prompt,
            "response": self.response
        }

class Observability:
    def __init__(self):
        # Langfuse client is always created after load_dotenv
        self.langfuse = Langfuse()

    def trace(self, user_query, retrieved_chunks, llm_prompt, llm_response):
        try:
            body = LangfuseTraceBody(user_query, retrieved_chunks, llm_prompt, llm_response)
            trace = self.langfuse.trace(body)
            if trace is not None:
                # Commented out span calls due to unsupported arguments in SDK v1.14.0
                # if hasattr(trace, "span"):
                #     try:
                #         trace.span(
                #             name="retrieval",
                #             input=user_query,
                #             output=retrieved_chunks
                #         )
                #         trace.span(
                #             name="generation",
                #             input=llm_prompt,
                #             output=llm_response
                #         )
                #     except Exception as span_e:
                #         print("Langfuse span error:", span_e)
                if hasattr(trace, "update"):
                    try:
                        trace.update(
                            output={"response": llm_response}
                        )
                    except Exception as update_e:
                        print("Langfuse update error:", update_e)
            self.langfuse.flush()
            return True
        except Exception as trace_e:
            print("Langfuse trace error:", trace_e)
            raise

if __name__ == "__main__":
    obs = Observability()
    obs.trace(
        "What is RAG?",
        ["chunk1", "chunk2"],
        "Prompt text",
        "LLM response"
    )