import pdfplumber
from unstructured.partition.pdf import partition_pdf
from PIL import Image
import os

# PDF Ingestion and Chunking
class PDFIngestor:
    def __init__(self, pdf_path):
        self.pdf_path = pdf_path

    def extract_text_tables_images(self):
        text_chunks = []
        table_chunks = []
        image_chunks = []
        with pdfplumber.open(self.pdf_path) as pdf:
            for i, page in enumerate(pdf.pages):
                # Extract text
                text = page.extract_text()
                if text:
                    text_chunks.append({'type': 'text', 'content': text, 'page': i+1})
                # Extract tables
                tables = page.extract_tables()
                for table in tables:
                    table_chunks.append({'type': 'table', 'content': table, 'page': i+1})
                # Extract images
                for img_idx, img in enumerate(page.images):
                    x0, top, x1, bottom = img['x0'], img['top'], img['x1'], img['bottom']
                    cropped = page.crop((x0, top, x1, bottom)).to_image(resolution=150)
                    img_path = f"data/page_{i+1}_img_{img_idx}.png"
                    cropped.save(img_path, format="PNG")
                    image_chunks.append({'type': 'image', 'path': img_path, 'page': i+1})
        return text_chunks, table_chunks, image_chunks

    def chunk_text(self, text, chunk_size=500, overlap=50):
        # Simple recursive chunking
        chunks = []
        start = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunks.append(text[start:end])
            start = end - overlap
        return chunks

# Example usage
if __name__ == "__main__":
    pdf_path = "data/sample.pdf"  # Place your PDF here
    ingestor = PDFIngestor(pdf_path)
    text_chunks, table_chunks, image_chunks = ingestor.extract_text_tables_images()
    # Chunk text further
    all_text_chunks = []
    for chunk in text_chunks:
        all_text_chunks.extend(ingestor.chunk_text(chunk['content']))
    print(f"Text chunks: {len(all_text_chunks)} | Tables: {len(table_chunks)} | Images: {len(image_chunks)}")
