
import os
from dotenv import load_dotenv
load_dotenv(dotenv_path="C:/Users/sahan/OneDrive - Singapore University of Technology and Design/Desktop/NCS-chatbot/.env", override=True)
print("[DEBUG] LANGFUSE_HOST after load_dotenv:", os.getenv("LANGFUSE_HOST"))
print("[DEBUG] LANGFUSE_SECRET_KEY after load_dotenv:", os.getenv("LANGFUSE_SECRET_KEY"))
print("[DEBUG] LANGFUSE_PUBLIC_KEY after load_dotenv:", os.getenv("LANGFUSE_PUBLIC_KEY"))

from fastapi import FastAPI, UploadFile, File, Form
from pydantic import BaseModel
from ingest.pdf_ingest import PDFIngestor
from embeddings.embed import TextTableEmbedder, ImageEmbedder
from retrieval.vectordb import VectorDB
from llm.llm import LLM
from observability.langfuse_client import get_langfuse
from observability.langfuse_obs import LangfuseTraceBody
import uuid
from langfuse import Langfuse
from langfuse.client import StatefulTraceClient
import tempfile
import torch
import clip

app = FastAPI()

class DummyBody:
    def __init__(self, input):
        self.input = input
        self.id = None
app = FastAPI()

@app.on_event("startup")
def startup_event():
    load_dotenv(override=True)
    print("[DEBUG] LANGFUSE_HOST in startup event:", os.getenv("LANGFUSE_HOST"))

class QueryRequest(BaseModel):
    query: str


@app.post("/chat")
async def chat(query: str = Form(...), pdf: UploadFile = File(...)):
    print("[DEBUG] LANGFUSE_HOST in /chat:", os.getenv("LANGFUSE_HOST"))
    # ...existing code...

    # Save uploaded PDF to a temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(await pdf.read())
        tmp_path = tmp.name

    # Ingest PDF
    ingestor = PDFIngestor(tmp_path)
    text_chunks, table_chunks, image_chunks = ingestor.extract_text_tables_images()
    text_embedder = TextTableEmbedder()
    image_embedder = ImageEmbedder()

    # Embed and index
    all_texts = [chunk['content'] for chunk in text_chunks]
    text_embs = text_embedder.embed(all_texts) if all_texts else []
    meta = text_chunks
    vectordb = VectorDB(text_embs.shape[1] if len(text_embs) > 0 else 384)
    if len(text_embs) > 0:
        vectordb.add(text_embs, meta)

    # Add tables
    if table_chunks:
        table_texts = [str(chunk['content']) for chunk in table_chunks]
        table_embs = text_embedder.embed(table_texts)
        vectordb.add(table_embs, table_chunks)

    # Create separate index for images
    image_db = None
    if image_chunks:
        img_emb_dim = 512  # CLIP ViT-B/32 default
        image_db = VectorDB(img_emb_dim)
        for chunk in image_chunks:
            img_emb = image_embedder.embed(chunk['path'])
            image_db.add([img_emb], [chunk])

    llm = LLM()

    # Embed query
    query_emb = text_embedder.embed([query])[0]
    # Retrieve from text/table index
    retrieved = vectordb.search(query_emb, top_k=5)

    # Retrieve from image index using CLIP text embedding
    image_retrieved = []
    if image_db:
        text_tokens = clip.tokenize([query]).to(image_embedder.device)
        with torch.no_grad():
            clip_query_emb = image_embedder.model.encode_text(text_tokens).cpu().numpy().flatten()
        image_retrieved = image_db.search(clip_query_emb, top_k=2)

    # Build context and prompt
    context = "\n".join([str(c['content']) for c in retrieved])
    prompt = f"Answer the following using context:\n{context}\nQuestion: {query}"

    response = llm.generate(prompt)

    # Now create the Langfuse trace with both prompt and response
    lf = get_langfuse()
    # Explicitly set input/output fields as dicts for Langfuse trace
    body = LangfuseTraceBody(
        input={"user_question": query},
        output={"answer": response},
        query=query,
        retrieved_chunks=[],
        prompt=prompt,
        response=response
    )
    trace = lf.trace(body)
    print("Trace type:", type(trace))

    # Embed and index
    all_texts = [chunk['content'] for chunk in text_chunks]
    text_embs = text_embedder.embed(all_texts) if all_texts else []
    meta = text_chunks
    vectordb = VectorDB(text_embs.shape[1] if len(text_embs) > 0 else 384)
    if len(text_embs) > 0:
        vectordb.add(text_embs, meta)

    # Add tables
    if table_chunks:
        table_texts = [str(chunk['content']) for chunk in table_chunks]
        table_embs = text_embedder.embed(table_texts)
        vectordb.add(table_embs, table_chunks)

    # Create separate index for images
    image_db = None
    if image_chunks:
        img_emb_dim = 512  # CLIP ViT-B/32 default
        image_db = VectorDB(img_emb_dim)
        for chunk in image_chunks:
            img_emb = image_embedder.embed(chunk['path'])
            image_db.add([img_emb], [chunk])

    llm = LLM()

    # Embed query
    query_emb = text_embedder.embed([query])[0]
    # Retrieve from text/table index
    retrieved = vectordb.search(query_emb, top_k=5)

    # Retrieve from image index using CLIP text embedding
    image_retrieved = []
    if image_db:
        text_tokens = clip.tokenize([query]).to(image_embedder.device)
        with torch.no_grad():
            clip_query_emb = image_embedder.model.encode_text(text_tokens).cpu().numpy().flatten()
        image_retrieved = image_db.search(clip_query_emb, top_k=2)

    # Build context
    context = "\n".join([str(c['content']) for c in retrieved])
    prompt = f"Answer the following using context:\n{context}\nQuestion: {query}"
    response = llm.generate(prompt)

    # Observability: trace with Langfuse
    lf.flush()

    # Clean up temp file
    os.remove(tmp_path)

    return {"answer": response, "retrieved": retrieved + image_retrieved, "trace_type": str(type(trace))}

