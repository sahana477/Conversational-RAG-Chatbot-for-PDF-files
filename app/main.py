from fastapi import FastAPI, UploadFile, File, Form
from pydantic import BaseModel
from ingest.pdf_ingest import PDFIngestor
from embeddings.embed import TextTableEmbedder, ImageEmbedder
from retrieval.vectordb import VectorDB
from llm.llm import LLM
from observability.langfuse_obs import Observability
import os
import tempfile
from dotenv import load_dotenv
import torch
load_dotenv()

app = FastAPI()

class QueryRequest(BaseModel):
    query: str

@app.post("/chat")
async def chat(query: str = Form(...), pdf: UploadFile = File(...)):
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
        import clip
        text_tokens = clip.tokenize([query]).to(image_embedder.device)
        with torch.no_grad():
            clip_query_emb = image_embedder.model.encode_text(text_tokens).cpu().numpy().flatten()
        image_retrieved = image_db.search(clip_query_emb, top_k=2)

    # Build context
    context = "\n".join([str(c['content']) for c in retrieved])
    prompt = f"Answer the following using context:\n{context}\nQuestion: {query}"
    response = llm.generate(prompt)

    # Clean up temp file
    os.remove(tmp_path)

    return {"answer": response, "retrieved": retrieved + image_retrieved}
