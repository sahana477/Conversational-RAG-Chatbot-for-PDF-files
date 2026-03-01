# RAG Chatbot over PDF with Langfuse

This project is a Python-based conversational Retrieval-Augmented Generation (RAG) chatbot that can answer questions over a PDF containing text, tables, and images. It uses FAISS for vector search, sentence-transformers for text/table embeddings, CLIP for image embeddings, and Langfuse for tracing and evaluation. FastAPI is used for the API interface.

## Features
- PDF ingestion (text, tables, images)
- Chunking and embedding
- Vector DB retrieval
- LLM integration (OpenAI/vLLM)
- Langfuse observability

## Setup
1. Place your PDF in the `data/` folder.
2. Install dependencies: `pip install -r requirements.txt`
3. Run the API: `uvicorn app.main:app --reload`

## Modules
- `ingest/`: PDF parsing and chunking
- `embeddings/`: Embedding models
- `retrieval/`: Vector DB and retrieval logic
- `llm/`: LLM integration
- `observability/`: Langfuse tracing
- `app/`: FastAPI app

## Demo
- Answer 3–5 queries via API or UI
- Show Langfuse traces

## Productionization
- Containerize with Docker
- Deploy vector DB as a service
- Use Langfuse for monitoring

---
Replace the sample PDF and update config as needed.