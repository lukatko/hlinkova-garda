# build_embeddings.py
import json
import os
import sys
from pathlib import Path
import PyPDF2
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.utils import embedding_functions
import numpy as np


# Config
PDF_FILES = ["data/annual_reports/Erste_Group_2024.pdf", "data/annual_reports/GSK_esg-performance-report_2023.pdf", "data/annual_reports/swisscom_sustainability_impact_report_2024_en.pdf"]
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
CHUNKS_FILE = "chunks.json"

# 1. Parse PDFs per page
def parse_pdf(pdf_path):
    parsed = []
    doc_name = Path(pdf_path).name
    with open(pdf_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for i, page in enumerate(reader.pages, start=1):
            text = page.extract_text()
            if text:
                parsed.append({"doc": doc_name, "page": i, "text": text})
    return parsed

# Chunk text while keeping doc + page
def chunk_text(text, doc, page, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    words = text.split()
    for i in range(0, len(words), size - overlap):
        yield {"doc": doc, "page": page, "text": " ".join(words[i:i+size])}

# Collect chunks from all PDFs
all_chunks = []
for pdf_file in PDF_FILES:
    parsed_pages = parse_pdf(pdf_file)
    for page_doc in parsed_pages:
        all_chunks.extend(chunk_text(page_doc["text"], page_doc["doc"], page_doc["page"]))

print(f"Total chunks: {len(all_chunks)}")

# Build embeddings
model = SentenceTransformer("all-MiniLM-L6-v2")
# Initialize Chroma client with persistent storage
client = chromadb.PersistentClient(path="./chroma_db")
# Get or create collection
collection = client.get_or_create_collection("pdf_chunks")
print(f"ChromaDB data will be stored in: {os.path.abspath('./chroma_db')}")

# ----------------------------
# 5. Add chunks to Chroma
# ----------------------------
print("Starting to add chunks to ChromaDB...")

for i, c in enumerate(all_chunks):
    try:
        # Generate embedding directly with the model
        embedding = model.encode(c["text"]).tolist()  # Convert numpy array to list
        
        # Add to collection
        collection.add(
            ids=[str(i)],
            metadatas=[{"doc": c["doc"], "page": c["page"]}],
            documents=[c["text"]],
            embeddings=[embedding]  # Pass as list
        )
        
        if i % 10 == 0:  # Print progress every 10 chunks
            print(f"Added chunk {i}/{len(all_chunks)}")
    except Exception as e:
        print(f"Error processing chunk {i}: {e}", file=sys.stderr)
        raise  # Re-raise the exception after logging
print(f"All chunks added to ChromaDB")

# ----------------------------
# 6. Save chunk metadata for reference
# ----------------------------
with open(CHUNKS_FILE, "w", encoding="utf-8") as f:
    json.dump(all_chunks, f, ensure_ascii=False, indent=2)

print(f"Chunk metadata saved to {CHUNKS_FILE}")