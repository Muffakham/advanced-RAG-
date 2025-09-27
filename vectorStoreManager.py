from typing import List, Optional
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from pathlib import Path

class VectorStoreManager:
    """
    A base class to manage the creation and persistence of a Chroma vector store.
    This class handles loading an existing vector store or creating a new one if needed.
    """

    def __init__(self, 
                 embeddings: OpenAIEmbeddings,
                 persist_directory: str,
                 collection_name: str,
                 force_reindex: bool = False,
                 splits: Optional[List[Document]] = None):
        """
        Initializes the VectorStoreManager.

        Args:
            embeddings (OpenAIEmbeddings): The embeddings model to use.
            persist_directory (str): The directory to persist the vector store.
            collection_name (str): The name of the collection in the vector store.
            force_reindex (bool, optional): Whether to force re-indexing. Defaults to False.
            splits (Optional[List[Document]], optional): A list of document splits to index. Defaults to None.
        """
        self.embeddings = embeddings
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.force_reindex = force_reindex
        self.vector_store = self._get_or_create_vector_store(splits)

    def _load_existing_store(self) -> Optional[Chroma]:
        """
        Tries to load an existing persisted vector store.

        Returns:
            Optional[Chroma]: The loaded vector store, or None if it doesn't exist or loading fails.
        """
        vector_store = None
        if not self.force_reindex and Path(self.persist_directory).exists():
            try:
                vector_store = Chroma(
                    persist_directory=self.persist_directory,
                    collection_name=self.collection_name,
                    embedding_function=self.embeddings,
                )
                print(f"Loaded existing Chroma store from {self.persist_directory} (collection: {self.collection_name})")
            except Exception as e:
                print(f"Could not load existing store, will reindex: {e}")
        return vector_store

    def _create_chroma_vector_store(self, splits: Optional[List[Document]]) -> Chroma:
        """
        Creates a new Chroma vector store.

        Args:
            splits (Optional[List[Document]]): The document splits to be indexed.

        Returns:
            Chroma: The newly created vector store.
        """
        if splits:
            vector_store = Chroma.from_documents(
                documents=splits, 
                embedding=self.embeddings,
                persist_directory=self.persist_directory,
                collection_name=self.collection_name
            )
            print(f"Knowledge base indexed with {len(splits)} chunks.")
        else:
            vector_store = Chroma(
                embedding_function=self.embeddings,
                persist_directory=self.persist_directory,
                collection_name=self.collection_name
            )
            print(f"Initialized a new, empty vector store at {self.persist_directory}.")
        
        vector_store.persist()
        return vector_store

    def _get_or_create_vector_store(self, splits: Optional[List[Document]]) -> Chroma:
        """
        Gets an existing vector store or creates a new one if it doesn't exist or if re-indexing is forced.

        Args:
            splits (Optional[List[Document]]): The document splits to be indexed if a new store is created.

        Returns:
            Chroma: The vector store.
        """
        vector_store = self._load_existing_store()
        if not vector_store:
            vector_store = self._create_chroma_vector_store(splits)
        return vector_store
    
    def get_retriever(self, k: int = 5):
        """
        Creates a retriever for the vector store.

        Args:
            k (int, optional): The number of documents to retrieve. Defaults to 5.

        Returns:
            VectorStoreRetriever: A retriever for the vector store.
        """
        return self.vector_store.as_retriever(search_kwargs={"k": k})
