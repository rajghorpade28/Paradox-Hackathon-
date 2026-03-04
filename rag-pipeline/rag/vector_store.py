import logging
from langchain_chroma import Chroma
from rag.embeddings import get_embeddings_model
from utils.config import EMBEDDINGS_DIR

logger = logging.getLogger(__name__)

_vector_store = None

def get_vector_store():
    """
    Lazy loads the Chroma vector store.
    """
    global _vector_store
    if _vector_store is None:
        logger.info(f"Loading Chroma database from {EMBEDDINGS_DIR}")
        embeddings = get_embeddings_model()
        _vector_store = Chroma(
            collection_name="rag_collection",
            embedding_function=embeddings,
            persist_directory=str(EMBEDDINGS_DIR)
        )
    return _vector_store

def add_documents_to_store(chunks):
    """
    Adds chunks to the vector store.
    """
    if not chunks:
        logger.warning("No chunks to add to vector store.")
        return
    
    store = get_vector_store()
    logger.info(f"Adding {len(chunks)} chunks to vector store.")
    store.add_documents(chunks)
