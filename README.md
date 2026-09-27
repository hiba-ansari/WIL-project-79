# 🛡️ Travel Insurance Policy AI Assistant

An intelligent Retrieval-Augmented Generation (RAG) system designed to simplify the complex process of understanding travel insurance Policy Disclosure Statements (PDS). This tool allows users to ask natural language questions and receive precise, evidence-backed answers extracted from actual insurance documents.

## 🚀 Key Features

- **Context-Aware QA**: Uses RAG to ensure answers are grounded in specific policy documents, minimizing LLM hallucinations.
- **Multi-Insurer Support**: Indexes and queries documents from **Allianz**, **Budget Direct**, **Cover-More**, and **Medibank**.
- **Local & Private**: Powered by [Ollama](https://ollama.com/), meaning the LLM and Embedding models run locally on your machine. Your data never leaves your environment.
- **Interactive UI**: A clean, user-friendly interface built with **Streamlit** for easy interaction and source verification.
- **Transparent Citations**: Every response includes specific source filenames and page numbers, allowing users to verify the AI's answer against the original PDF.

## 🛠️ Technical Architecture

The system follows a classic RAG pipeline:
1. **Ingestion**: PDFs $\rightarrow$ Text Extraction (`PyPDF`) $\rightarrow$ Recursive Chunking $\rightarrow$ Vector Embeddings (`nomic-embed-text`) $\rightarrow$ Vector Store (`ChromaDB`).
2. **Retrieval**: User Query $\rightarrow$ Embedding $\rightarrow$ Cosine Similarity Search in ChromaDB $\rightarrow$ Top-K relevant chunks.
3. **Generation**: System Prompt + Retrieved Context + User Query $\rightarrow$ LLM (`llama3`) $\rightarrow$ Grounded Answer.

## 📦 Installation & Setup

### 1. Prerequisites
Install **Ollama** from [ollama.com](https://ollama.com/download) and pull the required models:
```bash
ollama pull llama3
ollama pull nomic-embed-text
```

### 2. Environment Setup
Clone the repository and set up the Python environment:
```bash
# Navigate to the project folder
cd WIL-project-79

# Set up virtual environment
chmod +x setup.sh
./setup.sh
source .venv/bin/activate
```

### 3. Knowledge Base Indexing
Before running the app, you must index the insurance PDFs:
```bash
python src/ingest.py
```
*This will process the PDFs in `data/raw_docs/` and create a persistent vector database in `data/vector_db/`.*

## 💻 Running the Application

Start the Streamlit web interface:
```bash
streamlit run app.py
```
Once the app is running, open your browser to the local URL provided (usually `http://localhost:8501`).

## 📊 Evaluation Framework

The system is benchmarked using a golden dataset (`data/evaluation/qna_dataset.json`) across three critical dimensions:

| Category | Focus | Goal |
| :--- | :--- | :--- |
| **Factual** | Specific limits & dates | Test retrieval precision (e.g., "What is the laptop limit?"). |
| **Reasoning** | Scenario-based logic | Test LLM's ability to apply rules (e.g., "I rode a bike without a helmet; am I covered?"). |
| **Out-of-Knowledge** | Irrelevant queries | Test the system's ability to admit it doesn't know (e.g., "What's the weather in Tokyo?"). |

## 📂 Project Structure

```text
WIL-project-79/
├── app.py                # Streamlit Web Interface
├── requirements.txt      # Project dependencies
├── setup.sh              # Setup script for .venv
├── src/
│   ├── ingest.py        # Data ingestion & Vectorization pipeline
│   └── query.py          # RAG retrieval & generation logic
├── data/
│   ├── raw_docs/         # Source insurance PDF files
│   ├── vector_db/        # ChromaDB persistent storage
│   └── evaluation/       # Q&A benchmarking dataset
└── doc/
    └── Planning.md       # Design specifications
```

## 👥 Contributors
- Hiba Ansari
- Chanduru Ananthakumar
- Dipesh Shrestha
- Siris Sakhakarmi
