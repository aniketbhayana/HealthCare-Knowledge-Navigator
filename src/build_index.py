import fitz
import re
import json

from sentence_transformers import SentenceTransformer

# ======================
# Configuration
# ======================

PDF_PATH = "data/sample.pdf"
OUTPUT_PATH = "index.json"

CHUNK_SIZE = 800
CHUNK_OVERLAP = 150


# ======================
# Text Cleaning
# ======================

def clean_text(text):
    # Fix words split across lines
    text = re.sub(r'(\w)-\n(\w)', r'\1\2', text)

    # Replace remaining newlines with spaces
    text = text.replace("\n", " ")

    # Collapse multiple spaces
    text = re.sub(r"\s+", " ", text)

    return text.strip()


# ======================
# Chunk Creation
# ======================

def create_chunks(document):
    chunks = []

    for page_num in range(len(document)):

        text = document[page_num].get_text()
        text = clean_text(text)

        start = 0

        while start < len(text):

            chunk_text = text[start:start + CHUNK_SIZE]

            # Skip tiny chunks
            if len(chunk_text.strip()) >= 150:

                chunks.append({
                    "page": page_num + 1,
                    "text": chunk_text
                })

            start += CHUNK_SIZE - CHUNK_OVERLAP

    return chunks


# ======================
# Main
# ======================

print("Loading PDF...")

document = fitz.open(PDF_PATH)

print("Creating chunks...")

chunks = create_chunks(document)

print(f"Chunks created: {len(chunks)}")

print("Loading embedding model...")

model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

index = []

print("Generating embeddings...")

for chunk_id, chunk in enumerate(chunks):

    embedding = model.encode(
        chunk["text"]
    )

    indexed_chunk = {
        "id": chunk_id,
        "source": "sample.pdf",
        "page": chunk["page"],
        "text": chunk["text"],
        "embedding": embedding.tolist()
    }

    index.append(indexed_chunk)

    # Progress indicator
    if chunk_id % 100 == 0:
        print(f"Processed {chunk_id}/{len(chunks)} chunks")

print("Saving index...")

with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(index, f)

print(f"Done. Saved {len(index)} chunks to {OUTPUT_PATH}")