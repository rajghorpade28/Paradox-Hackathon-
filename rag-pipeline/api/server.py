import os
import shutil
import logging
from typing import List, Dict, Any
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager

from utils.config import DATA_DIR
from rag.chunking import chunk_documents
from rag.vector_store import add_documents_to_store
from rag.retriever import retrieve_and_rerank
from rag.generator import generate_answer

# Local imports for specific loading
from langchain_community.document_loaders import PyPDFLoader, TextLoader, CSVLoader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting RAG FastAPI Server...")
    # Lazy initializations will trigger on first requests to save startup time
    yield
    logger.info("Shutting down RAG FastAPI Server...")

app = FastAPI(title="RAG Hackathon API", lifespan=lifespan)

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    answer: str
    sources: List[str]
    confidence: float

@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.post("/upload-documents")
async def upload_documents(files: List[UploadFile] = File(...)):
    """
    Saves uploaded files to disk, chunks them, and adds to VectorStore.
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")
        
    saved_files = []
    
    # Save files
    for file in files:
        try:
            file_path = DATA_DIR / file.filename
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            saved_files.append(file.filename)
            logger.info(f"Saved file {file.filename}")
        except Exception as e:
            logger.error(f"Failed to save {file.filename}: {e}")
            raise HTTPException(status_code=500, detail=f"Could not save file {file.filename}")
            
    # Process files
    logger.info("Processing newly uploaded documents...")
    try:
        docs = []
        for file_name in saved_files:
            file_path = DATA_DIR / file_name
            if file_path.suffix.lower() == '.pdf':
                loader = PyPDFLoader(str(file_path))
                d = loader.load()
                for x in d: x.metadata['source_file'] = file_name
                docs.extend(d)
            elif file_path.suffix.lower() == '.txt':
                loader = TextLoader(str(file_path), encoding='utf-8')
                d = loader.load()
                for x in d: x.metadata['source_file'] = file_name
                docs.extend(d)
            elif file_path.suffix.lower() == '.csv':
                loader = CSVLoader(str(file_path))
                d = loader.load()
                for x in d: x.metadata['source_file'] = file_name
                docs.extend(d)
                
        chunks = chunk_documents(docs)
        add_documents_to_store(chunks)
        
        return {
            "message": f"Successfully uploaded and processed {len(saved_files)} files.",
            "files": saved_files,
            "chunks_added": len(chunks)
        }
    except Exception as e:
        logger.error(f"Error processing documents: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query", response_model=QueryResponse)
def query_rag(request: QueryRequest):
    """
    1. Retrieves top K chunks.
    2. Reranks to top N.
    3. Generates answer.
    """
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Empty query")
        
    try:
        context_docs = retrieve_and_rerank(request.query)
        result = generate_answer(request.query, context_docs)
        
        return QueryResponse(
            answer=result["answer"],
            sources=result["sources"],
            confidence=result["confidence"]
        )
    except Exception as e:
        logger.error(f"Error during query processing: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error during query")
