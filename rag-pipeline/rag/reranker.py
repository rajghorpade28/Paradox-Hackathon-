import logging
import torch
from sentence_transformers import CrossEncoder
from utils.config import RERANKER_MODEL_NAME

logger = logging.getLogger(__name__)

_reranker = None

def get_reranker():
    """
    Lazy loads the CrossEncoder for BAAI/bge-reranker-large.
    """
    global _reranker
    if _reranker is None:
        logger.info(f"Loading reranker model: {RERANKER_MODEL_NAME}")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _reranker = CrossEncoder(RERANKER_MODEL_NAME, max_length=512, device=device)
    return _reranker

def rerank_documents(query: str, documents: list, top_n: int = 4):
    """
    Reranks langchain documents based on a query.
    """
    if not documents:
        return []

    ranker = get_reranker()
    
    # Prepare sentence pairs
    pairs = [[query, doc.page_content] for doc in documents]
    
    # Predict scores
    scores = ranker.predict(pairs)
    
    # Pair scores with documents and sort
    doc_score_pairs = list(zip(documents, scores))
    doc_score_pairs.sort(key=lambda x: x[1], reverse=True)
    
    # Select top_n
    top_docs = [doc for doc, score in doc_score_pairs[:top_n]]
    
    return top_docs
