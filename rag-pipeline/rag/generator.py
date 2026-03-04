import os
import logging
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from rag.prompt import get_prompt
from utils.config import LLM_MODEL_NAME

logger = logging.getLogger(__name__)

_llm = None

def get_llm():
    """
    Lazy loads the OpenAI LLM.
    """
    global _llm
    if _llm is None:
        logger.info(f"Loading LLM: {LLM_MODEL_NAME}")
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key or api_key == "your_openai_api_key_here":
            logger.warning("OPENAI_API_KEY appears to be missing or invalid in environment.")
        _llm = ChatOpenAI(model=LLM_MODEL_NAME, temperature=0.0)
    return _llm

def generate_answer(query: str, context_docs: list):
    """
    Generates an answer using the LLM with the provided context.
    Returns Answer, Source Citations, and a pseudo Confidence Score.
    """
    if not context_docs:
        return {
            "answer": "I cannot find the answer. Please provide relevant context.",
            "sources": [],
            "confidence": 0.0
        }
        
    # Build context string
    context_texts = []
    sources = set()
    for i, doc in enumerate(context_docs):
        # Extract filename if possible, otherwise use source
        source_val = doc.metadata.get('source_file', doc.metadata.get('source', f'Document_{i}'))
        # Normalize in case source is a full path
        source_name = os.path.basename(source_val) if isinstance(source_val, str) else str(source_val)
        sources.add(source_name)
        context_texts.append(f"--- Document Source: {source_name} ---\n{doc.page_content}")
        
    context_string = "\n\n".join(context_texts)
    
    prompt = get_prompt()
    chain = prompt | get_llm() | StrOutputParser()
    
    logger.info("Calling LLM for generation...")
    try:
        response = chain.invoke({
            "retrieved_documents": context_string,
            "user_query": query
        })
        # Basic confidence estimation heuristic based on length and missing info
        if "cannot find the answer" in response.lower():
            confidence = 0.1
        else:
            confidence = 0.95
            
    except Exception as e:
        logger.error(f"LLM Generation failed: {e}")
        response = f"Sorry, an error occurred during generation: {e}"
        confidence = 0.0
        
    return {
        "answer": response,
        "sources": list(sources),
        "confidence": confidence
    }
