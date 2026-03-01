import faiss
import numpy as np

class VectorDB:
    def __init__(self, dim):
        self.index = faiss.IndexFlatL2(dim)
        self.vectors = []
        self.metadata = []
    def add(self, embeddings, metadatas):
        self.index.add(np.array(embeddings).astype('float32'))
        self.vectors.extend(embeddings)
        self.metadata.extend(metadatas)
    def search(self, query_emb, top_k=5):
        D, I = self.index.search(np.array([query_emb]).astype('float32'), top_k)
        results = []
        for idx in I[0]:
            results.append(self.metadata[idx])
        return results

# Example usage
if __name__ == "__main__":
    # Dummy data
    emb_dim = 384
    db = VectorDB(emb_dim)
    dummy_embs = np.random.rand(10, emb_dim)
    dummy_meta = [{'type': 'text', 'content': f'Chunk {i}'} for i in range(10)]
    db.add(dummy_embs, dummy_meta)
    query_emb = np.random.rand(emb_dim)
    results = db.search(query_emb, top_k=3)
    print("Top results:", results)
