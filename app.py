import os
import streamlit as st

from src.rag_engine import ask_question

# =====================
# Page Config
# =====================

st.set_page_config(
    page_title="Clinical Guideline Navigator",
    page_icon="🏥",
    layout="wide"
)

# =====================
# Sidebar
# =====================

with st.sidebar:

    st.title(
        "🏥 Clinical Guideline Navigator"
    )

    st.divider()

    st.subheader(
        "📚 Current Knowledge Sources"
    )

    try:

        pdf_files = [
            file
            for file in os.listdir("data")
            if file.endswith(".pdf")
        ]

        for pdf in pdf_files:

            st.write(
                f"📄 {pdf.replace('.pdf', '')}"
            )

    except:

        st.write(
            "No guideline documents found."
        )

    st.divider()

    st.subheader("ℹ️ About")

    st.info(
        """
        Clinical Guideline Navigator is a
        Retrieval Augmented Generation (RAG)
        system that answers healthcare
        questions using trusted clinical
        guidelines.
        """
    )

    st.markdown(
        """
        **Powered By**

        • Gemini 2.5 Flash

        • ChromaDB

        • Sentence Transformers

        • Streamlit
        """
    )

    st.divider()

    st.subheader("💡 Tips")

    st.markdown(
        """
        • Ask follow-up questions

        • Sources are shown below answers

        • Answers are grounded in the
          loaded guideline documents

        • Use specific medical questions
          for best results
        """
    )

    st.divider()

    if st.button(
        "🗑️ Clear Chat",
        use_container_width=True
    ):

        st.session_state.messages = []

        st.rerun()

# =====================
# Main Header
# =====================

st.title(
    "🏥 Clinical Guideline Navigator"
)

st.caption(
    "Evidence based answers from trusted clinical guidelines"
)

# =====================
# Chat History
# =====================

if "messages" not in st.session_state:

    st.session_state.messages = []

# =====================
# Display Messages
# =====================

for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )

        if (
            message["role"] == "assistant"
            and "sources" in message
        ):

            with st.expander(
                "📚 Sources"
            ):

                for source in message["sources"]:

                    st.write(
                        f"• {source}"
                    )

# =====================
# Chat Input
# =====================

prompt = st.chat_input(
    "Ask a healthcare question..."
)

if prompt:

    # User Message

    st.session_state.messages.append({
        "role": "user",
        "content": prompt
    })

    with st.chat_message("user"):

        st.markdown(prompt)

    # Assistant Message

    with st.chat_message(
        "assistant"
    ):

        with st.spinner(
            "Searching guidelines..."
        ):

            answer, sources = ask_question(
                prompt
            )

        st.markdown(answer)

        with st.expander(
            "📚 Sources"
        ):

            for source in sources:

                st.write(
                    f"• {source}"
                )

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources
    })