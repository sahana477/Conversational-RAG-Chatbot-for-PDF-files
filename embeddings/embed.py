from sentence_transformers import SentenceTransformer
import numpy as np
from PIL import Image
import torch
import clip

# Text/Table Embedding
class TextTableEmbedder:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
    def embed(self, texts):
        return self.model.encode(texts, show_progress_bar=False)

# Image Embedding
class ImageEmbedder:
    def __init__(self, device="cpu"):
        self.device = device
        self.model, self.preprocess = clip.load("ViT-B/32", device)
    def embed(self, image_path):
        image = Image.open(image_path).convert("RGB")
        image_input = self.preprocess(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            image_features = self.model.encode_image(image_input)
        return image_features.cpu().numpy().flatten()

# Example usage
if __name__ == "__main__":
    texts = ["Example text chunk 1", "Example text chunk 2"]
    text_embedder = TextTableEmbedder()
    text_embs = text_embedder.embed(texts)
    print(f"Text embeddings shape: {text_embs.shape}")
    img_embedder = ImageEmbedder()
    img_emb = img_embedder.embed("data/sample_image.png")  # Replace with actual image path
    print(f"Image embedding shape: {img_emb.shape}")
