from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from vectorStoreManager import VectorStoreManager
from graphState import GraphState

class MemoryManager(VectorStoreManager):
    """
    Manages the long-term conversational memory vector store.
    This class is responsible for storing older messages from the chat history
    into a persistent vector store to maintain long-term context.
    """

    def __init__(self, embeddings: OpenAIEmbeddings, persist_directory: str = "./vector_stores/memory"):
        """
        Initializes the MemoryManager.

        Args:
            embeddings (OpenAIEmbeddings): The embeddings model to use for vectorizing messages.
            persist_directory (str, optional): The directory to persist the long-term memory
                                             vector store. Defaults to "./vector_stores/memory".
        """
        super().__init__(
            embeddings=embeddings,
            persist_directory=persist_directory,
            collection_name="long_term_memory"
        )
    
    def store(self, state: GraphState) -> dict:
        """
        Stores older messages from the chat history into the long-term memory vector store
        and trims the short-term chat history.

        Args:
            state (GraphState): The current state of the graph, containing the chat history.

        Returns:
            dict: A dictionary with the updated (trimmed) chat history.
        """
        print("---STORING TO LONG-TERM MEMORY---")
        chat_history = state.get('chat_history', [])
        
        # If chat history exceeds a certain length, store older messages
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
        
        # Persist the changes to the vector store
        self.vector_store.persist()
        return {"chat_history": chat_history}
