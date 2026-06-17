import fitz
import re

from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim

CHUNK_SIZE = 800
CHUNK_OVERLAP = 150

model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

document = fitz.open("data/sample.pdf")

chunks = []

for page_num in range(len(document)):

    text = document[page_num].get_text()

    text = re.sub(r'(\w)-\n(\w)', r'\1\2', text)
    text = text.replace("\n", " ")
    text = re.sub(r'\s+', ' ', text)

    start = 0

    while start < len(text):

        chunk_text = text[start:start + CHUNK_SIZE]

        chunks.append({
            "page": page_num + 1,
            "text": chunk_text
        })

        start += CHUNK_SIZE - CHUNK_OVERLAP

print(f"Chunks loaded: {len(chunks)}")

query = "What eating patterns are recommended for diabetes?"

query_embedding = model.encode(query)

results = []

for chunk in chunks:

    chunk_embedding = model.encode(chunk["text"])

    score = cos_sim(
        query_embedding,
        chunk_embedding
    ).item()

    results.append({
        "score": score,
        "page": chunk["page"],
        "text": chunk["text"]
    })

results.sort(
    key=lambda x: x["score"],
    reverse=True
)

for result in results[:3]:

    print("\n===================")
    print(f"Score: {result['score']:.4f}")
    print(f"Page: {result['page']}")
    print(result["text"][:500])