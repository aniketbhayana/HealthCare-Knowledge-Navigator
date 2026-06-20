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

try:

    collection = chroma_client.get_collection(
        "healthcare_knowledge"
    )

except:

    from src.build_chroma import build_database

    build_database()

    collection = chroma_client.get_collection(
        "healthcare_knowledge"
    )

# -----------------------
# RAG Function
# -----------------------

def ask_question(question):

    query_embedding = embedding_model.encode(
        question
    ).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=5
    )

    context = ""
    sources = []

    for i in range(len(results["documents"][0])):

        page = results["metadatas"][0][i]["page"]

        source = results["metadatas"][0][i]["source"]

        text = results["documents"][0][i]

        context += f"\n[{source} | Page {page}]\n{text}\n"

        sources.append(
            f"{source} - Page {page}"
        )
    history = ""

    try:

        import streamlit as st

        recent_messages = (
            st.session_state.messages[-8:]
        )

        for msg in recent_messages:

            history += (
                f"{msg['role']}: "
                f"{msg['content']}\n"
            )

    except:

        pass

    prompt = f"""
    You are a healthcare knowledge navigator.

    Your job is to answer questions ONLY using the provided context.

    Rules:

    1. Do not use outside medical knowledge.
    2. If the answer is not present in the context,
       say:
       "I could not find that information in the provided guideline."
    3.Provide structured answers using bullet points when appropriate.Summarize key recommendations clearly.Do not quote large passages verbatim.
        Always present actionable recommendations first.
    4. Never invent recommendations or treatments.

    
    Conversation History:
    {history}

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

        except Exception:

            time.sleep(2)

    if answer is None:

        answer = (
            "Gemini is currently unavailable."
        )

    return answer, sorted(set(sources))