"""
University Admission Assistant
Streamlit Frontend for RAG Application
"""

import streamlit as st
from rag import generate_answer


# ---------------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------------

st.set_page_config(
    page_title="University Admission Assistant",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------
# CUSTOM CSS
# ---------------------------------------------------------

st.markdown(
    """
    <style>
        .main-title {
            font-size: 2.5rem;
            font-weight: 700;
            margin-bottom: 0.2rem;
        }

        .subtitle {
            font-size: 1.1rem;
            color: #666;
            margin-bottom: 1.5rem;
        }

        .source-box {
            padding: 10px;
            border-radius: 8px;
            border: 1px solid #ddd;
            margin-top: 10px;
        }

        .welcome-box {
            padding: 20px;
            border-radius: 12px;
            border: 1px solid #ddd;
            margin-bottom: 20px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# SESSION STATE
# ---------------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []


# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------

with st.sidebar:

    st.title("🎓 University Assistant")

    st.markdown("---")

    st.markdown(
        """
        ### About

        This AI assistant uses **Retrieval-Augmented Generation (RAG)**
        to answer questions from the university knowledge base.

        ### Knowledge Base

        - 📄 Fee Structure
        - 🏠 Hostel Rules
        - 🎓 Scholarship Policy
        - 💼 Placement information
        - 📚 Admission information

        ### Technology

        - Python
        - Streamlit
        - LangChain
        - ChromaDB
        - HuggingFace Embeddings
        - Google Gemini Api Key
        """
    )

    st.markdown("---")

    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# ---------------------------------------------------------
# MAIN HEADER
# ---------------------------------------------------------

st.markdown(
    '<div class="main-title">🎓 University Admission Assistant</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="subtitle">'
    "Ask questions about fees, scholarships, hostel, admissions and more."
    "</div>",
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# WELCOME MESSAGE
# ---------------------------------------------------------

if not st.session_state.messages:

    st.markdown(
        """
        <div class="welcome-box">

        ### 👋 Welcome!

        I can help you find information from the university
        knowledge base.

        **Try asking:**

        - What is the tuition fee for Computer Science Engineering?
        - What is the hostel fee?
        - What scholarships are available?
        - What are the scholarship eligibility criteria?
        - What is the admission fee?
        - What are the hostel rules?
        - What is the payment schedule?

        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------
# DISPLAY CHAT HISTORY
# ---------------------------------------------------------

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])

        if message.get("sources"):

            with st.expander("📚 View Sources"):

                for source in message["sources"]:
                    st.markdown(f"- `{source}`")


# ---------------------------------------------------------
# USER INPUT
# ---------------------------------------------------------

user_question = st.chat_input(
    "Ask a question about university admissions..."
)


# ---------------------------------------------------------
# PROCESS QUESTION
# ---------------------------------------------------------

if user_question:

    # Display user message
    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_question,
        }
    )

    with st.chat_message("user"):
        st.markdown(user_question)

    # Generate answer
    with st.chat_message("assistant"):

        with st.spinner("🔎 Searching the knowledge base..."):

            try:

                result = generate_answer(user_question)

                # -----------------------------------------
                # HANDLE DIFFERENT RETURN TYPES
                # -----------------------------------------

                answer = ""
                sources = []

                if isinstance(result, str):

                    answer = result

                elif isinstance(result, dict):

                    # Try common keys
                    answer = (
                        result.get("answer")
                        or result.get("text")
                        or result.get("response")
                        or ""
                    )

                    sources = result.get("sources", [])

                else:

                    # LangChain AIMessage / similar object
                    if hasattr(result, "content"):
                        answer = result.content
                    else:
                        answer = str(result)

                # -----------------------------------------
                # CLEAN RAW GEMINI RESPONSE
                # -----------------------------------------

                if isinstance(answer, list):

                    text_parts = []

                    for item in answer:

                        if isinstance(item, dict):
                            text_parts.append(
                                item.get("text", str(item))
                            )
                        else:
                            text_parts.append(str(item))

                    answer = "\n".join(text_parts)

                answer = str(answer)

                # Remove accidental wrappers
                if answer.startswith("{'type': 'text'"):
                    try:
                        import ast

                        parsed = ast.literal_eval(answer)

                        if isinstance(parsed, dict):
                            answer = parsed.get("text", answer)

                    except Exception:
                        pass

                # -----------------------------------------
                # DISPLAY ANSWER
                # -----------------------------------------

                st.markdown(answer)

                # -----------------------------------------
                # DISPLAY SOURCES
                # -----------------------------------------

                if sources:

                    with st.expander("📚 View Sources"):

                        for source in sources:
                            st.markdown(f"- `{source}`")

                # -----------------------------------------
                # SAVE ASSISTANT MESSAGE
                # -----------------------------------------

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                        "sources": sources,
                    }
                )

            except Exception as e:

                error_message = (
                    "❌ Sorry, I couldn't process that question.\n\n"
                    f"**Error:** `{str(e)}`"
                )

                st.error(error_message)

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": error_message,
                        "sources": [],
                    }
                )

