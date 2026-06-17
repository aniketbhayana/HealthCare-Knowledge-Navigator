import os
import re
import fitz
import chromadb

from sentence_transformers import SentenceTransformer

# ======================
# Configuration
# ======================

DATA_DIR = "data"

CHUNK_SIZE = 800
CHUNK_OVERLAP = 150

COLLECTION_NAME = "healthcare_knowledge"


# ======================
# Text Cleaning
# ======================

def clean_text(text):

    # Fix words split across lines
    text = re.sub(
        r'(\w)-\n(\w)',
        r'\1\2',
        text
    )

    # Replace line breaks
    text = text.replace("\n", " ")

    # Remove extra spaces
    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# ======================
# Chunking
# ======================

def create_chunks(document):

    chunks = []

    for page_num in range(len(document)):

        text = document[page_num].get_text()

        text = clean_text(text)

        start = 0

        while start < len(text):

            chunk_text = text[
                start:start + CHUNK_SIZE
            ]

            if len(chunk_text.strip()) >= 150:

                chunks.append({
                    "page": page_num + 1,
                    "text": chunk_text
                })

            start += (
                CHUNK_SIZE - CHUNK_OVERLAP
            )

    return chunks


# ======================
# Find PDFs
# ======================

pdf_files = [
    file
    for file in os.listdir(DATA_DIR)
    if file.endswith(".pdf")
]

print(f"Found {len(pdf_files)} PDFs")

for pdf in pdf_files:
    print(f" - {pdf}")


# ======================
# Embedding Model
# ======================

print("\nLoading embedding model...")

model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)


# ======================
# Chroma Setup
# ======================

print("Connecting to Chroma...")

client = chromadb.PersistentClient(
    path="./chroma_db"
)

try:
    client.delete_collection(
        name=COLLECTION_NAME
    )
    print("Old collection deleted")
except:
    print("No existing collection found")

collection = client.create_collection(
    name=COLLECTION_NAME
)

print(
    f"Created collection: {COLLECTION_NAME}"
)


# ======================
# Process PDFs
# ======================

total_chunks = 0

for pdf_file in pdf_files:

    print(
        f"\nProcessing {pdf_file}"
    )

    document = fitz.open(
        os.path.join(
            DATA_DIR,
            pdf_file
        )
    )

    chunks = create_chunks(
        document
    )

    print(
        f"Created {len(chunks)} chunks"
    )

    total_chunks += len(chunks)

    for chunk_id, chunk in enumerate(chunks):

        embedding = model.encode(
            chunk["text"]
        ).tolist()

        chunk_unique_id = (
            f"{pdf_file}_{chunk_id}"
        )

        collection.add(
            ids=[chunk_unique_id],
            documents=[
                chunk["text"]
            ],
            metadatas=[{
                "page": chunk["page"],
                "source": pdf_file
            }],
            embeddings=[
                embedding
            ]
        )

        if chunk_id % 100 == 0:

            print(
                f"Processed "
                f"{chunk_id}/{len(chunks)}"
            )


# ======================
# Done
# ======================

print("\n====================")
print("INGESTION COMPLETE")
print("====================")

print(
    f"Total PDFs: {len(pdf_files)}"
)

print(
    f"Total Chunks: {total_chunks}"
)

print(
    f"Collection: {COLLECTION_NAME}"
)