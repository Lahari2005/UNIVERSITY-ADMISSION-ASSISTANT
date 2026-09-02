# 🎓 University Admission Assistant

An AI-powered University Admission Assistant that uses **Retrieval-Augmented Generation (RAG)** to answer university admission-related questions from trusted documents.

## 🚀 Overview

The University Admission Assistant allows users to ask questions about university admissions and receive relevant, context-aware answers.

Instead of relying only on an LLM's general knowledge, the system retrieves information from a university-specific knowledge base and uses it to generate grounded responses.

## ✨ Key Features

- 📄 Processes university admission documents

- 🔍 Retrieves relevant information using semantic search

- 🤖 Generates answers using an LLM

- 🧠 Implements a Retrieval-Augmented Generation (RAG) pipeline

- 💬 Provides an interactive user interface

- 📚 Uses university-specific information as the knowledge base

## 🏗️ Architecture

```text

University Documents

        ↓

Document Ingestion

        ↓

Text Chunking

        ↓

Embedding Generation

        ↓

ChromaDB Vector Store

        ↓

User Question

        ↓

Similarity Search

        ↓

Relevant Context

        ↓

LLM

        ↓

Generated Answer
 🛠️ Technologies Used

Python

LangChain

ChromaDB

Sentence Transformers

Streamlit

Large Language Models (LLMs)

Retrieval-Augmented Generation (RAG)

📁 Project Structure
 
UNIVERSITY-ADMISSION-ASSISTANT/

│

├── data/

│   └── University admission documents

│

├── app.py

├── ingest.py

├── rag.py

├── utils.py

├── requirements.txt

├── .gitignore

└── README.md
 
⚙️ Installation

Clone the repository:
git clone https://github.com/Lahari2005/UNIVERSITY-ADMISSION-ASSISTANT.git
GitHub - Lahari2005/UNIVERSITY-ADMISSION-ASSISTANT
Contribute to Lahari2005/UNIVERSITY-ADMISSION-ASSISTANT development by creating an account on GitHub.
 
Navigate to the project:
cd UNIVERSITY-ADMISSION-ASSISTANT
 
Create and activate a virtual environment:
python -m venv .venv
 
Install the required dependencies:
pip install -r requirements.txt
 
🔐 Environment Variable:
Create a .env file and add the required API credentials:
OPENAI_API_KEY=your_api_key
 
▶️ Running the Project
First, process the documents:
python ingest.py
 
Then run the Streamlit application:
streamlit run app.py

The application will open in your browser.

🔄 RAG Pipeline
The project follows these main steps:
Document Ingestion – Admission documents are loaded and processed.
Text Processing – Documents are split into smaller chunks.
Embedding Generation – Text chunks are converted into vector embeddings.
Vector Storage – Embeddings are stored in ChromaDB.
Retrieval – Relevant chunks are retrieved based on the user’s question.
Generation – Retrieved context is provided to the LLM to generate an answer.

🎯 Use Case
This project demonstrates how Generative AI and RAG can be used to build a domain-specific question-answering system for university admissions.

👩‍💻 Author
Lahari Pakalapati
 
 