import streamlit as st

from src.rag_engine import ask_question

st.set_page_config(
    page_title="Clinical Guideline Navigator",
    page_icon="🏥",
    layout="wide"
)

st.title("🏥 Clinical Guideline Navigator")
st.caption(
    "Evidence-based answers from ADA, WHO and NICE guidelines"
)

# =====================
# Chat History
# =====================

if "messages" not in st.session_state:
    st.session_state.messages = []

# =====================
# Display Chat History
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
                "Sources"
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

    # Show user message
    st.session_state.messages.append({
        "role": "user",
        "content": prompt
    })

    with st.chat_message("user"):

        st.markdown(prompt)

    # Generate answer
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
            "Sources"
        ):

            for source in sources:

                st.write(
                    f"• {source}"
                )

    # Save assistant response
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources
    })