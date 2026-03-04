import logging
from langchain_huggingface import HuggingFaceEmbeddings
from utils.config import EMBEDDING_MODEL_NAME

logger = logging.getLogger(__name__)

_embeddings_model = None

def get_embeddings_model():
    """
    Lazy load the embedding model to save resources.
    Returns the HuggingFace bge-small-en model.
    """
    global _embeddings_model
    if _embeddings_model is None:
        logger.info(f"Loading embedding model: {EMBEDDING_MODEL_NAME}")
        _embeddings_model = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            model_kwargs={'device': 'cpu'}, 
            encode_kwargs={'normalize_embeddings': True} 
        )
    return _embeddings_model
