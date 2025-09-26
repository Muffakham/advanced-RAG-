
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from vectorStoreManager import VectorStoreManager
from graphState import GraphState

class MemoryManager(VectorStoreManager):
    """Manages the long-term conversational memory vector store."""

    def __init__(self, embeddings: OpenAIEmbeddings, persist_directory: str = "./vector_stores/memory"):
        super().__init__(
            embeddings=embeddings,
            persist_directory=persist_directory,
            collection_name="long_term_memory"
        )
    
    def store(self, chat_history: GraphState):
        """
        Stores older messages into the long-term memory.
        Returns the trimmed chat history.
        """
        print("---STORING TO LONG-TERM MEMORY---")
        
        if len(chat_history) > 10:
            messages_to_store = chat_history[:-10]
            docs_to_store = [
                Document(page_content=f"{msg.type}: {msg.content}")
                for msg in messages_to_store
            ]
            self.vector_store.add_documents(docs_to_store)
            print(f"Stored {len(docs_to_store)} messages to long-term memory.")
            
            # Trim the short-term chat history to the last 10 messages
            chat_history = chat_history[-10:]
            print("Short-term chat history trimmed to last 10 messages.")
        
        self.vector_store.persist()
        return chat_history