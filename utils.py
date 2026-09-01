"""
utils.py

Shared configuration and utility functions for the
University Admission Assistant.
"""

import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings


# ---------------------------------------------------------
# Environment
# ---------------------------------------------------------

load_dotenv()


# ---------------------------------------------------------
# Project Paths
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"

UPLOAD_DIR = BASE_DIR / "uploads"

CHROMA_DIR = BASE_DIR / "chroma_db"

LOG_DIR = BASE_DIR / "logs"


# Create required directories
DATA_DIR.mkdir(exist_ok=True)

UPLOAD_DIR.mkdir(exist_ok=True)

CHROMA_DIR.mkdir(exist_ok=True)

LOG_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------
# Chroma Configuration
# ---------------------------------------------------------

COLLECTION_NAME = "university_admission"


# ---------------------------------------------------------
# Embedding Model
# ---------------------------------------------------------

EMBEDDING_MODEL_NAME = (
    "sentence-transformers/all-MiniLM-L6-v2"
)


# ---------------------------------------------------------
# Logging
# ---------------------------------------------------------

LOG_FILE = LOG_DIR / "app.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(
            LOG_FILE,
            encoding="utf-8"
        ),
        logging.StreamHandler(),
    ],
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------
# Embedding Model
# ---------------------------------------------------------

def get_embedding_model() -> HuggingFaceEmbeddings:
    """
    Create the HuggingFace embedding model.

    The same embedding model is used for:
    1. Semantic chunking
    2. ChromaDB vector embeddings
    """

    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME
    )


# ---------------------------------------------------------
# Semantic Chunker
# ---------------------------------------------------------

def get_semantic_chunker() -> SemanticChunker:
    """
    Create a SemanticChunker using the same
    HuggingFace embedding model.
    """

    embedding_model = get_embedding_model()

    return SemanticChunker(
        embeddings=embedding_model,
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=95,
    )


# ---------------------------------------------------------
# ChromaDB
# ---------------------------------------------------------

def get_vector_store() -> Chroma:
    """
    Return the persistent ChromaDB vector store.
    """

    embedding_model = get_embedding_model()

    return Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=str(CHROMA_DIR),
        embedding_function=embedding_model,
    )


# ---------------------------------------------------------
# Gemini API Key
# ---------------------------------------------------------

def get_google_api_key() -> str:
    """
    Read Google Gemini API key from .env.
    """

    api_key = os.getenv("GOOGLE_API_KEY")

    if not api_key:
        raise ValueError(
            "GOOGLE_API_KEY is missing. "
            "Add it to your .env file."
        )

    return api_key