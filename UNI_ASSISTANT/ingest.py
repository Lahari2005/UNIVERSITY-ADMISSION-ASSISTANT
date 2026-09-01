"""
ingest.py

University Admission Assistant
Document ingestion pipeline.

Supported:
    - PDF
    - TXT

Pipeline:

Document
    ↓
PyPDFLoader / TextLoader
    ↓
Metadata
    ↓
SemanticChunker
    ↓
HuggingFace Embeddings
    ↓
ChromaDB
"""

from pathlib import Path
import shutil
import sys

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
)

from utils import (
    DATA_DIR,
    CHROMA_DIR,
    COLLECTION_NAME,
    get_semantic_chunker,
    get_vector_store,
    logger,
)


# ---------------------------------------------------------
# Supported file extensions
# ---------------------------------------------------------

SUPPORTED_EXTENSIONS = {
    ".pdf",
    ".txt",
}


# ---------------------------------------------------------
# Load document
# ---------------------------------------------------------

def load_document(file_path: Path):
    """Load a PDF or TXT document."""

    extension = file_path.suffix.lower()

    logger.info(
        "Loading document: %s",
        file_path.name,
    )

    if extension == ".pdf":

        loader = PyPDFLoader(
            str(file_path)
        )

    elif extension == ".txt":

        loader = TextLoader(
            str(file_path),
            encoding="utf-8",
            autodetect_encoding=True,
        )

    else:

        raise ValueError(
            f"Unsupported file type: {extension}"
        )

    documents = loader.load()

    logger.info(
        "Loaded %d document pages/sections from %s",
        len(documents),
        file_path.name,
    )

    return documents


# ---------------------------------------------------------
# Add metadata
# ---------------------------------------------------------

def add_document_metadata(
    documents,
    file_path: Path,
):
    """Add and preserve document metadata."""

    document_type = (
        file_path.suffix
        .lower()
        .replace(".", "")
    )

    for index, document in enumerate(documents):

        document.metadata["source"] = str(
            file_path
        )

        document.metadata["filename"] = (
            file_path.name
        )

        # PyPDFLoader gives zero-based page numbers.
        # Convert to human-readable page numbers.

        if "page" in document.metadata:

            document.metadata["page"] = (
                int(document.metadata["page"]) + 1
            )

        else:

            document.metadata["page"] = None

        document.metadata["document_type"] = (
            document_type
        )

        document.metadata["loader_index"] = index

    return documents


# ---------------------------------------------------------
# Semantic chunking
# ---------------------------------------------------------

def create_semantic_chunks(documents):
    """
    Split documents using LangChain SemanticChunker.

    RecursiveCharacterTextSplitter is NOT used.
    """

    logger.info(
        "Creating semantic chunks..."
    )

    chunker = get_semantic_chunker()

    chunks = chunker.split_documents(
        documents
    )

    logger.info(
        "Created %d semantic chunks",
        len(chunks),
    )

    return chunks


# ---------------------------------------------------------
# Chunk IDs
# ---------------------------------------------------------

def assign_chunk_ids(chunks):
    """Assign unique IDs to every chunk."""

    for index, chunk in enumerate(chunks):

        filename = chunk.metadata.get(
            "filename",
            "unknown",
        )

        page = chunk.metadata.get(
            "page",
            "unknown",
        )

        chunk.metadata["chunk_id"] = (
            f"{filename}_page_{page}_chunk_{index}"
        )

    return chunks


# ---------------------------------------------------------
# Process one file
# ---------------------------------------------------------

def process_file(file_path: Path):
    """Process one PDF/TXT file."""

    logger.info(
        "Processing: %s",
        file_path.name,
    )

    documents = load_document(
        file_path
    )

    documents = add_document_metadata(
        documents,
        file_path,
    )

    chunks = create_semantic_chunks(
        documents
    )

    chunks = assign_chunk_ids(
        chunks
    )

    logger.info(
        "Finished %s -> %d chunks",
        file_path.name,
        len(chunks),
    )

    return chunks


# ---------------------------------------------------------
# Find supported files
# ---------------------------------------------------------

def get_supported_files():
    """Find PDF and TXT files inside data/."""

    if not DATA_DIR.exists():

        logger.warning(
            "Data directory does not exist: %s",
            DATA_DIR,
        )

        return []

    files = [
        file
        for file in DATA_DIR.iterdir()
        if (
            file.is_file()
            and file.suffix.lower()
            in SUPPORTED_EXTENSIONS
        )
    ]

    return sorted(files)


# ---------------------------------------------------------
# Delete existing ChromaDB
# ---------------------------------------------------------

def delete_chroma_database():
    """Delete the existing ChromaDB directory."""

    if CHROMA_DIR.exists():

        logger.info(
            "Deleting existing ChromaDB..."
        )

        shutil.rmtree(
            CHROMA_DIR,
            ignore_errors=True,
        )

    CHROMA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    logger.info(
        "ChromaDB directory recreated."
    )


# ---------------------------------------------------------
# Rebuild knowledge base
# ---------------------------------------------------------

def rebuild_knowledge_base():
    """
    Rebuild the entire ChromaDB knowledge base.
    """

    logger.info(
        "========== KNOWLEDGE BASE REBUILD STARTED =========="
    )

    files = get_supported_files()

    if not files:

        print(
            "\nNo PDF or TXT files found."
        )

        print(
            f"Place your documents inside:\n{DATA_DIR}"
        )

        return False

    print(
        f"\nFound {len(files)} document(s)."
    )

    # Delete old database
    delete_chroma_database()

    all_chunks = []

    # Process every document
    for file_number, file_path in enumerate(
        files,
        start=1,
    ):

        print(
            f"\n[{file_number}/{len(files)}] "
            f"Processing: {file_path.name}"
        )

        try:

            chunks = process_file(
                file_path
            )

            all_chunks.extend(
                chunks
            )

            print(
                f"    Created {len(chunks)} chunks"
            )

        except Exception as error:

            logger.exception(
                "Failed to process %s",
                file_path.name,
            )

            print(
                f"    ERROR: {error}"
            )

    if not all_chunks:

        print(
            "\nERROR: No chunks were created."
        )

        return False

    # Generate embeddings and store in ChromaDB
    print(
        "\nGenerating embeddings and storing "
        "chunks in ChromaDB..."
    )

    try:

        vector_store = get_vector_store()

        vector_store.add_documents(
            documents=all_chunks
        )

    except Exception as error:

        logger.exception(
            "Failed to store documents in ChromaDB."
        )

        print(
            f"\nChromaDB ERROR: {error}"
        )

        return False

    print(
        "\n========================================"
    )

    print(
        "KNOWLEDGE BASE CREATED SUCCESSFULLY"
    )

    print(
        "========================================"
    )

    print(
        f"Documents : {len(files)}"
    )

    print(
        f"Chunks    : {len(all_chunks)}"
    )

    print(
        f"Database  : {CHROMA_DIR}"
    )

    print(
        f"Collection: {COLLECTION_NAME}"
    )

    logger.info(
        "Knowledge base rebuilt successfully."
    )

    return True


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

if __name__ == "__main__":

    try:

        success = rebuild_knowledge_base()

        if not success:

            sys.exit(1)

    except KeyboardInterrupt:

        print(
            "\nProcess cancelled by user."
        )

        sys.exit(1)

    except Exception as error:

        logger.exception(
            "Unexpected ingestion error."
        )

        print(
            f"\nUnexpected error: {error}"
        )

        sys.exit(1)