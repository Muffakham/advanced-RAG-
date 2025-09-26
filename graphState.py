from typing import List, TypedDict
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage


class GraphState(TypedDict):
    """
    Represents the state of our graph.
    
    Attributes:
        chat_history: The conversation history (short-term memory).
        question: The most recent user question.
        sub_queries: A list of deconstructed sub-queries from the main question.
        retrieved_documents: The combined documents from both knowledge base and long-term conversation memory.
    """
    chat_history: List[BaseMessage]
    question: str
    sub_queries: List[str]
    retrieved_documents: List[Document]