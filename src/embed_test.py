from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim

model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

sentences = [
    "What is diabetes?",
    "Explain diabetes treatment",
    "How to bake a cake?"
]

embeddings = model.encode(sentences)

similarity_ab = cos_sim(
    embeddings[0],
    embeddings[1]
)

similarity_ac = cos_sim(
    embeddings[0],
    embeddings[2]
)

print("Diabetes vs Treatment")
print(similarity_ab.item())

print("\nDiabetes vs Cake")
print(similarity_ac.item())