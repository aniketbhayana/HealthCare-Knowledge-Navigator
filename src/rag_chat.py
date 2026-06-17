import os
import time

from dotenv import load_dotenv
from google import genai

import chromadb
from sentence_transformers import SentenceTransformer

# -----------------------
# Setup
# -----------------------

load_dotenv()

client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

chroma_client = chromadb.PersistentClient(
    path="./chroma_db"
)

collection = chroma_client.get_collection(
    "healthcare_knowledge"
)

# -----------------------
# Query
# -----------------------

while True:

    question = input("\nAsk: ")

    if question.lower() == "exit":
        print("Goodbye!")
        break

    query_embedding = embedding_model.encode(
        question
    ).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=3
    )

    context = ""
    sources = []

    for i in range(len(results["documents"][0])):

        page = results["metadatas"][0][i]["page"]

        text = results["documents"][0][i]

        context += f"\n[Page {page}]\n{text}\n"

        sources.append(page)

    prompt = f"""
    You are a healthcare knowledge navigator.

    Your job is to answer questions ONLY using the
    provided context.

    Rules:

    1. Do not use outside medical knowledge.
    2. If the answer is not present in the context,
   say:
   "I could not find that information in the provided guideline."
    3. Be concise and factual.
    4. When possible, cite the page numbers mentioned in the context.
    5. Never invent recommendations or treatments.

    Context:
    {context}

    Question:
    {question}
    """

    answer = None

    for attempt in range(3):

        try:

            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt
            )

            answer = response.text
            break

        except Exception as e:

            print(
                f"Attempt {attempt+1} failed:{e}"
            )

            time.sleep(2)

    if answer is None:
        answer = "Gemini is currently unavailable."

    print("\n====================")
    print("ANSWER")
    print("====================\n")

    print(answer)

    print("\n====================")
    print("SOURCES")
    print("====================")

    for page in sorted(set(sources)):
        print(f"Page {page}")