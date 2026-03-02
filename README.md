# RAG Chatbot over PDF with Langfuse

This project is a Python-based conversational Retrieval-Augmented Generation (RAG) chatbot that can answer questions over a PDF containing text, tables, and images. It uses FAISS for vector search, sentence-transformers for text/table embeddings, CLIP for image embeddings, and Langfuse for tracing and evaluation. FastAPI is used for the API interface.

## Features
- PDF ingestion (text, tables, images)
- Chunking and embedding
- Vector DB retrieval
- LLM integration (OpenAI/vLLM)
- Langfuse observability

## Setup & Run Instructions

### 1. Clone the Repository
```powershell
git clone https://github.com/sahana477/Conversational-RAG-Chatbot-for-PDF-files.git
cd Conversational-RAG-Chatbot-for-PDF-files
```

### 2. Create and Configure Environment
- Copy `.env.example` to `.env` and fill in your secrets (Langfuse, HuggingFace, etc).
```powershell
copy .env.example .env
# Edit .env with your credentials
```

### 3. Install Python Dependencies
```powershell
pip install -r requirements.txt
```

### 4. Run FastAPI Backend
```powershell
uvicorn app.main:app --reload
```

### 5. Run Streamlit UI
```powershell
streamlit run app/streamlit_ui.py
```

### 6. Upload a PDF and Start Chatting
- Open the Streamlit UI in your browser (usually http://localhost:8501)
- Upload a PDF and enter your query.

---
**Note:**
- Ensure your `.env` file is present and correct before running.
- If you encounter errors, check your environment variables and Python version (recommended: Python 3.9+).
- For Langfuse tracing, verify your Langfuse credentials and dashboard.

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