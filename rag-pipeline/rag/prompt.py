from langchain.prompts import PromptTemplate

RAG_PROMPT_TEMPLATE = """You are an expert assistant.

Answer the question strictly using the provided context.

If the answer is not present in the context, say you cannot find the answer.

Context:
{retrieved_documents}

Question:
{user_query}
"""

def get_prompt():
    return PromptTemplate(
        template=RAG_PROMPT_TEMPLATE,
        input_variables=["retrieved_documents", "user_query"]
    )
