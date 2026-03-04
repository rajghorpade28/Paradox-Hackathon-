import logging
from rag.vector_store import get_vector_store
from rag.reranker import rerank_documents
from utils.config import TOP_K_INITIAL, TOP_K_FINAL

logger = logging.getLogger(__name__)

def retrieve_and_rerank(query: str, top_k_initial: int = TOP_K_INITIAL, top_k_final: int = TOP_K_FINAL):
    """
    1. Retrieve top_k_initial chunks using semantic search.
    2. Rerank them to get top_k_final chunks.
    """
    store = get_vector_store()
    
    logger.info(f"Retrieving top {top_k_initial} chunks for query: '{query}'")
    initial_docs = store.similarity_search(query, k=top_k_initial)
    
    if not initial_docs:
        logger.warning("No documents found in initial retrieval.")
        return []
        
    logger.info(f"Reranking documents to top {top_k_final}")
    final_docs = rerank_documents(query, initial_docs, top_n=top_k_final)
    
    return final_docs
