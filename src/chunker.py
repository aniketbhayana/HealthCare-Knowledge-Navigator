import fitz
import re

CHUNK_SIZE = 800
CHUNK_OVERLAP = 150

document = fitz.open("data/sample.pdf")

chunks = []

for page_num in range(len(document)):

    text = document[page_num].get_text()

    text = re.sub(r'(\w)-\n(\w)', r'\1\2', text)
    text = text.replace("\n", " ")
    text = re.sub(r'\s+', ' ', text)

    start = 0

    while start < len(text):

        end = start + CHUNK_SIZE

        chunk_text = text[start:end]

        chunks.append({
            "page": page_num + 1,
            "text": chunk_text
        })

        start += CHUNK_SIZE - CHUNK_OVERLAP

print(f"Total chunks: {len(chunks)}")

print("\nFirst chunk:\n")
print(chunks[0]["text"])

print("\nChunk metadata:\n")
print(chunks[0]["page"])