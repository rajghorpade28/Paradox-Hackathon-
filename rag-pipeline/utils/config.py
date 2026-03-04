import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Base directories
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data" / "documents"
EMBEDDINGS_DIR = BASE_DIR / "embeddings"

# Ensure directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)

# Chunking Configuration
CHUNK_SIZE = 700
CHUNK_OVERLAP = 100

# Models
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en"
RERANKER_MODEL_NAME = "BAAI/bge-reranker-large"
LLM_MODEL_NAME = "gpt-4o-mini" # Using 4o-mini as equivalent to 4.1-mini

# Retrieval Configuration
TOP_K_INITIAL = 10
TOP_K_FINAL = 4

# Server Configuration
HOST = "0.0.0.0"
PORT = 8000
