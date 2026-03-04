# RAG Hackathon Project

A production-ready, modular Retrieval Augmented Generation (RAG) system built for AI hackathons. The system features a robust FastAPI backend and a clean Streamlit frontend. It processes PDF, TXT, and CSV documents, stores them locally using ChromaDB, re-ranks the context using BAAI's CrossEncoder, and generates intelligent answers powered by GPT-4 models.

## Architecture

1. **User Query** Input through the Streamlit UI.
2. **Query Processing** Handled by FastAPI (`/query` endpoint).
3. **Embedding Generation** Chunked and embedded via `BAAI/bge-small-en`.
4. **Vector Database Search** Semantic similarity via ChromaDB (Top 10 chunks).
5. **Reranking Layer** Re-ranked using `BAAI/bge-reranker-large` (Top 4 chunks).
6. **Context Builder** Context aggregation in `rag/generator.py`.
7. **LLM Generation** Output string generation with GPT-4 (via LangChain).

## Tech Stack
* Python 3.10+
* FastAPI & Uvicorn (Backend)
* Streamlit (Frontend)
* LangChain Core (Orchestrator)
* HuggingFace Transformers (Embeddings & Reranking)
* ChromaDB (Local Vector Store)
* OpenAI GPT-4 API (LLM)

## Project Structure
```text
rag-hackathon/
├── data/
│   └── documents/          # Uploaded source files
├── embeddings/             # Chroma database persist location
├── rag/
│   ├── loader.py           # PyPDF, Text, and CSV Loaders
│   ├── chunking.py         # RecursiveCharacterTextSplitter
│   ├── embeddings.py       # BAAI/bge-small-en model loader
│   ├── vector_store.py     # ChromaDB wrapper
│   ├── retriever.py        # Pipeline for initial search
│   ├── reranker.py         # BAAI/bge-reranker-large cross-encoder
│   ├── prompt.py           # System & Generation Prompts
│   └── generator.py        # LLM Invocation
├── api/
│   └── server.py           # FastAPI server
├── app/
│   └── streamlit_app.py    # Streamlit frontend app
├── utils/
│   └── config.py           # Static configurations
├── main.py                 # FastAPI runner
├── requirements.txt
├── README.md
└── .env.example
```

## Installation Steps

1. **Clone & Enter the Repository**
   ```bash
   cd rag-hackathon
   ```
2. **Create a Virtual Environment (Optional but recommended)**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```
3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```
4. **Configure Environment variables**
   Copy `.env.example` to `.env` and assign your OpenAI API key:
   ```bash
   cp .env.example .env
   # Edit .env and set OPENAI_API_KEY
   ```

## Getting Started

To run the full stack, you need two terminal windows actively running the backend and the frontend.

**Terminal 1: Start the Backend (FastAPI)**
```bash
python main.py
```
*The API will be available at http://127.0.0.1:8000. You can view Swagger documentation at http://127.0.0.1:8000/docs.*

**Terminal 2: Start the Frontend (Streamlit)**
```bash
streamlit run app/streamlit_app.py
```

## How to use

1. Open the UI shown by the `streamlit run` command on your browser (usually `localhost:8501`).
2. **Upload Documents** in the sidebar. Select your target document(s) and click `Process Documents`. This locally saves your documents, creates overlapping vector chunks, and generates embeddings persisted in ChromaDB.
3. Wait for the upload sequence to finish successfully and the backend health to report "healthy".
4. Provide an intelligent **Question** related to your uploaded data inside the "Ask Questions" bar.
5. Hit `Ask Assistant` and watch the system perform retrieval, re-ranking, and high-fidelity text generation with verifiable sources!

*(Note: First-time runs will pull hugging-face models `bge-small-en` and `bge-reranker-large` directly to your local cache - this might take 1-3 minutes depending on bandwidth).*
