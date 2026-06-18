import streamlit as st

from src.rag_engine import ask_question

st.set_page_config(
    page_title="Healthcare Knowledge Navigator",
    page_icon="🏥",
    layout="wide"
)

# --------------------
# Sidebar
# --------------------

with st.sidebar:

    st.title("🏥 About")

    st.markdown("""
    **Healthcare Knowledge Navigator**

    Evidence based medical
    question answering using:

    - ChromaDB
    - Sentence Transformers
    - Gemini 2.5 Flash
    - Clinical Guidelines
    """)

    st.divider()

    st.markdown("""
    **Current Knowledge Base**

    • ADA Standards of Care

    • WHO 
                
    • NICE Diabetes Guidelines 2026

    • KDIGO 2026 Guidelines         
    
    """)

# --------------------
# Header
# --------------------

st.title(
    "🏥 Healthcare Knowledge Navigator"
)

st.caption(
    "Ask evidence-based questions from clinical guidelines"
)

st.divider()

# --------------------
# Input
# --------------------

question = st.text_area(
    "Clinical Question",
    placeholder="What eating patterns are recommended for diabetes management?",
    height=120
)

# --------------------
# Search
# --------------------

col1, col2, col3 = st.columns([1,1,1])

with col2:

    ask = st.button(
        "Ask Question",
        use_container_width=True
    )

# --------------------
# Results
# --------------------

if ask and question:

    with st.spinner(
        "Searching clinical guidelines..."
    ):

        answer, sources = ask_question(
            question
        )

    st.divider()

    st.subheader("💡 Answer")

    st.markdown("### 💡 Answer")
    st.markdown(answer)

    with st.expander("📚 Sources"):

        for source in sources:
            st.markdown(f"- {source}")