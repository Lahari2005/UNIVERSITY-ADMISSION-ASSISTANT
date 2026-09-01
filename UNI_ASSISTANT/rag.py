"""
rag.py

Retrieval-Augmented Generation engine for the
University Admission Assistant.

Pipeline:

User Question
      ↓
ChromaDB Similarity Search
      ↓
Top 4 Relevant Chunks
      ↓
Gemini 2.5 Flash
      ↓
Grounded Answer
      ↓
Source Citations
"""

from typing import Any

from langchain_google_genai import ChatGoogleGenerativeAI

from utils import (
    get_google_api_key,
    get_vector_store,
    logger,
)


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

MODEL_NAME = "gemini-3.5-flash"

TOP_K = 4


# ---------------------------------------------------------
# System Prompt
# ---------------------------------------------------------

SYSTEM_PROMPT = """
You are a University Admission Assistant.

Your job is to answer questions using ONLY the
retrieved university documents provided to you.

STRICT RULES:

1. Use ONLY the retrieved context.
2. Never answer from your own knowledge.
3. Never hallucinate.
4. If the answer is not present in the retrieved
   documents, reply exactly:

"I couldn't find this information in the university documents."

5. Always cite the source document filename(s) used
   to generate the answer.
6. Do not invent university policies, fees, deadlines,
   eligibility requirements, scholarships, or admission
   information.
7. If the retrieved documents contain conflicting
   information, clearly mention the conflict and cite
   the relevant source documents.
8. Keep the answer clear and concise.
9. When numerical information is available, preserve
   the exact values from the documents.
"""


# ---------------------------------------------------------
# Gemini
# ---------------------------------------------------------

def get_llm():
    """
    Create the Gemini 2.5 Flash model.
    """

    api_key = get_google_api_key()

    llm = ChatGoogleGenerativeAI(
        model=MODEL_NAME,
        google_api_key=api_key,
        temperature=0,
        max_retries=2,
    )

    return llm


# ---------------------------------------------------------
# Retrieve Documents
# ---------------------------------------------------------

def retrieve_documents(
    question: str,
    k: int = TOP_K,
):
    """
    Retrieve the most relevant documents from ChromaDB.

    Uses similarity search.

    Returns:
        List of documents with metadata.
    """

    if not question or not question.strip():
        raise ValueError(
            "Question cannot be empty."
        )

    logger.info(
        "Retrieving documents for question: %s",
        question,
    )

    vector_store = get_vector_store()

    documents = vector_store.similarity_search(
        question,
        k=k,
    )

    logger.info(
        "Retrieved %d documents",
        len(documents),
    )

    return documents


# ---------------------------------------------------------
# Retrieve With Scores
# ---------------------------------------------------------

def retrieve_documents_with_scores(
    question: str,
    k: int = TOP_K,
):
    """
    Retrieve documents along with similarity scores.

    Chroma returns distance values rather than
    traditional similarity percentages.

    Lower distance generally indicates greater
    similarity.
    """

    if not question or not question.strip():
        raise ValueError(
            "Question cannot be empty."
        )

    vector_store = get_vector_store()

    results = vector_store.similarity_search_with_score(
        question,
        k=k,
    )

    logger.info(
        "Retrieved %d documents with scores",
        len(results),
    )

    return results


# ---------------------------------------------------------
# Format Retrieved Context
# ---------------------------------------------------------

def format_context(documents) -> str:
    """
    Convert retrieved documents into a context string
    that can be passed to Gemini.
    """

    if not documents:
        return ""

    context_parts = []

    for index, document in enumerate(
        documents,
        start=1,
    ):

        metadata = document.metadata

        filename = metadata.get(
            "filename",
            "Unknown document",
        )

        page = metadata.get(
            "page",
            "Unknown page",
        )

        chunk_id = metadata.get(
            "chunk_id",
            "Unknown chunk",
        )

        content = document.page_content.strip()

        context_parts.append(
            f"""
--- Retrieved Document {index} ---

Source: {filename}
Page: {page}
Chunk ID: {chunk_id}

Content:
{content}

--- End Document {index} ---
"""
        )

    return "\n".join(
        context_parts
    )


# ---------------------------------------------------------
# Extract Sources
# ---------------------------------------------------------

def extract_sources(documents) -> list[str]:
    """
    Extract unique source filenames from retrieved
    documents.
    """

    sources = []

    for document in documents:

        filename = document.metadata.get(
            "filename"
        )

        if filename and filename not in sources:
            sources.append(filename)

    return sources


# ---------------------------------------------------------
# Build Prompt
# ---------------------------------------------------------

def build_prompt(
    question: str,
    context: str,
) -> str:
    """
    Build the final prompt sent to Gemini.
    """

    return f"""
{SYSTEM_PROMPT}

================ RETRIEVED CONTEXT ================

{context}

================ END CONTEXT ======================

USER QUESTION:

{question}

=====================================================

Answer the question using ONLY the retrieved context.

Remember:

- Do not use outside knowledge.
- Do not hallucinate.
- If the information is missing, use the exact
  fallback response specified in the system instructions.
- Always cite the source filename(s).
"""


# ---------------------------------------------------------
# Generate Answer
# ---------------------------------------------------------

def generate_answer(
    question: str,
) -> dict[str, Any]:
    """
    Complete RAG pipeline.

    Steps:

    1. Retrieve top 4 documents.
    2. Build context.
    3. Send context + question to Gemini.
    4. Extract source filenames.
    5. Return answer and retrieved documents.
    """

    if not question or not question.strip():

        raise ValueError(
            "Please enter a question."
        )

    logger.info(
        "Starting RAG query: %s",
        question,
    )

    # --------------------------------------------------
    # Retrieval
    # --------------------------------------------------

    retrieved_documents = (
        retrieve_documents(
            question,
            k=TOP_K,
        )
    )

    # --------------------------------------------------
    # No documents
    # --------------------------------------------------

    if not retrieved_documents:

        logger.warning(
            "No documents found for question."
        )

        return {
            "answer": (
                "I couldn't find this information "
                "in the university documents."
            ),
            "sources": [],
            "documents": [],
        }

    # --------------------------------------------------
    # Context
    # --------------------------------------------------

    context = format_context(
        retrieved_documents
    )

    # --------------------------------------------------
    # Prompt
    # --------------------------------------------------

    prompt = build_prompt(
        question,
        context,
    )

    # --------------------------------------------------
    # Gemini
    # --------------------------------------------------

    llm = get_llm()

    try:

        response = llm.invoke(
            prompt
        )

    except Exception as error:

        logger.exception(
            "Gemini API request failed."
        )

        raise RuntimeError(
            f"Gemini API request failed: {error}"
        ) from error

    # --------------------------------------------------
    # Response text
    # --------------------------------------------------

    answer = response.content

    if isinstance(answer, list):

        answer = "".join(
            str(item)
            for item in answer
        )

    answer = str(answer).strip()

    # --------------------------------------------------
    # Sources
    # --------------------------------------------------

    sources = extract_sources(
        retrieved_documents
    )

    logger.info(
        "RAG response generated successfully."
    )

    return {
        "answer": answer,
        "sources": sources,
        "documents": retrieved_documents,
    }


# ---------------------------------------------------------
# Simple Question Function
# ---------------------------------------------------------

def ask_question(
    question: str,
) -> str:
    """
    Simple interface that returns only the answer.
    """

    result = generate_answer(
        question
    )

    return result["answer"]


# ---------------------------------------------------------
# Debug Retrieval
# ---------------------------------------------------------

def debug_retrieval(
    question: str,
):
    """
    Print retrieved documents and scores.

    Useful during development.
    """

    results = retrieve_documents_with_scores(
        question,
        k=TOP_K,
    )

    print(
        "\n========== RETRIEVED DOCUMENTS ==========\n"
    )

    for index, (
        document,
        score,
    ) in enumerate(
        results,
        start=1,
    ):

        print(
            f"Document {index}"
        )

        print(
            f"Score: {score}"
        )

        print(
            f"Filename: "
            f"{document.metadata.get('filename')}"
        )

        print(
            f"Page: "
            f"{document.metadata.get('page')}"
        )

        print(
            f"Chunk ID: "
            f"{document.metadata.get('chunk_id')}"
        )

        print(
            "\nContent:"
        )

        print(
            document.page_content[:1000]
        )

        print(
            "\n-----------------------------------------\n"
        )


# ---------------------------------------------------------
# Main Test
# ---------------------------------------------------------

if __name__ == "__main__":

    print(
        "University Admission Assistant - RAG Test"
    )

    question = input(
        "\nEnter your question: "
    )

    try:

        result = generate_answer(
            question
        )

        print(
            "\n========== ANSWER ==========\n"
        )

        print(
            result["answer"]
        )

        print(
            "\n========== SOURCES ==========\n"
        )

        for source in result["sources"]:

            print(
                f"- {source}"
            )

    except Exception as error:

        print(
            f"\nERROR: {error}"
        )
