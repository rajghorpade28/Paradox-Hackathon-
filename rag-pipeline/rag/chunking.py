from langchain.text_splitter import RecursiveCharacterTextSplitter
from utils.config import CHUNK_SIZE, CHUNK_OVERLAP

def get_text_splitter():
    """
    Returns a configured RecursiveCharacterTextSplitter.
    """
    return RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        is_separator_regex=False,
    )

def chunk_documents(documents):
    """
    Splits a list of Document objects into chunks, preserving metadata.
    """
    if not documents:
        return []
    text_splitter = get_text_splitter()
    return text_splitter.split_documents(documents)
