import chromadb

from sentence_transformers import SentenceTransformer

model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

client = chromadb.PersistentClient(
    path="./chroma_db"
)

collection = client.get_collection(
    "diabetes_guidelines"
)

query = "What eating patterns are recommended for diabetes?"

query_embedding = model.encode(
    query
).tolist()

results = collection.query(
    query_embeddings=[query_embedding],
    n_results=3
)

results = collection.query(
    query_embeddings=[query_embedding],
    n_results=3
)

for i in range(3):

    page = results["metadatas"][0][i]["page"]
    score = results["distances"][0][i]
    text = results["documents"][0][i]

    print("\n================")
    print(f"Result #{i+1}")
    print(f"Page: {page}")
    print(f"Distance: {score:.4f}")
    print(text[:500])