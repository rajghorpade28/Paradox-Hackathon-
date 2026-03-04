import os
import logging
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader, TextLoader, CSVLoader
from utils.config import DATA_DIR

logger = logging.getLogger(__name__)

def load_documents(directory_path: str | Path = DATA_DIR):
    """
    Load all PDF, TXT, and CSV documents from the given directory.
    """
    directory_path = Path(directory_path)
    documents = []
    
    if not directory_path.exists():
        logger.warning(f"Directory {directory_path} does not exist. Creating it.")
        directory_path.mkdir(parents=True, exist_ok=True)
        return documents

    for file_path in directory_path.iterdir():
        if file_path.is_file():
            try:
                if file_path.suffix.lower() == '.pdf':
                    loader = PyPDFLoader(str(file_path))
                    docs = loader.load()
                    for d in docs: d.metadata['source_file'] = file_path.name
                    documents.extend(docs)
                    logger.info(f"Loaded {file_path}")
                elif file_path.suffix.lower() == '.txt':
                    loader = TextLoader(str(file_path), encoding='utf-8')
                    docs = loader.load()
                    for d in docs: d.metadata['source_file'] = file_path.name
                    documents.extend(docs)
                    logger.info(f"Loaded {file_path}")
                elif file_path.suffix.lower() == '.csv':
                    loader = CSVLoader(str(file_path))
                    docs = loader.load()
                    for d in docs: d.metadata['source_file'] = file_path.name
                    documents.extend(docs)
                    logger.info(f"Loaded {file_path}")
                else:
                    logger.debug(f"Skipping unsupported file format: {file_path.name}")
            except Exception as e:
                logger.error(f"Error loading {file_path}: {e}")

    return documents
