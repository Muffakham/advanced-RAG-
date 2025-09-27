from typing import List
from langchain_core.documents import Document
from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.retrievers.ensemble import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from pathlib import Path

from vectorStoreManager import VectorStoreManager

class KnowledgeBaseManager(VectorStoreManager):
    """
    Manages the creation and access of the knowledge base vector store.
    This class is responsible for loading documents from a specified directory,
    splitting them into chunks, and creating a hybrid retriever that combines
    vector-based search with BM25 keyword search.
    """
    
    def __init__(self, 
                 embeddings: OpenAIEmbeddings = None,
                 kb_path: str = "kb",
                 persist_directory: str = "./vector_stores/knowledge_base",
                 collection_name: str = "rag_knowledge",
                 chunk_size: int = 1000,
                 chunk_overlap: int = 200,
                 force_reindex: bool = False):
        """
        Initializes the KnowledgeBaseManager.

        Args:
            embeddings (OpenAIEmbeddings, optional): Embeddings model. Defaults to None.
            kb_path (str, optional): Path to the knowledge base directory. Defaults to "kb".
            persist_directory (str, optional): Directory to persist the vector store. Defaults to "./vector_stores/knowledge_base".
            collection_name (str, optional): Name of the collection in the vector store. Defaults to "rag_knowledge".
            chunk_size (int, optional): Size of text chunks. Defaults to 1000.
            chunk_overlap (int, optional): Overlap between text chunks. Defaults to 200.
            force_reindex (bool, optional): Whether to force re-indexing of the knowledge base. Defaults to False.
        """
        
        self.embeddings = embeddings or OpenAIEmbeddings()
        self.kb_path = Path(kb_path)
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.force_reindex = force_reindex
        
        # Load and split documents
        docs = self._load_documents_from_kb()
        self.splits = self._split_documents(docs)

        # Initialize the parent VectorStoreManager to create the vector store
        super().__init__(
            embeddings=self.embeddings,
            persist_directory=persist_directory,
            collection_name=collection_name,
            force_reindex=force_reindex,
            splits=self.splits
        )
    
    def _load_documents_from_kb(self) -> List[Document]:
        """
        Loads all .txt files from the specified knowledge base path.

        Returns:
            List[Document]: A list of loaded documents.
        """
        print(f"Indexing knowledge base from {self.kb_path}/ directory...")
        docs: List[Document] = []

        if self.kb_path.exists():
            txt_files = sorted(self.kb_path.glob("*.txt"))
            if not txt_files:
                print("No .txt files found. Creating an empty vector store.")
            for p in txt_files:
                try:
                    file_docs = TextLoader(str(p)).load()
                    docs.extend(file_docs)
                except Exception as e:
                    print(f"Skipping {p.name}: {e}")
        else:
            print(f"{self.kb_path}/ directory not found. Creating an empty vector store.")
        
        return docs

    def _split_documents(self, docs: List[Document]) -> List[Document]:
        """
        Splits documents into chunks using RecursiveCharacterTextSplitter.

        Args:
            docs (List[Document]): The documents to be split.

        Returns:
            List[Document]: A list of document chunks.
        """
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size, 
            chunk_overlap=self.chunk_overlap
        )
        return text_splitter.split_documents(docs) if docs else []

    def get_retriever(self, k: int = 5):
        """
        Creates a hybrid retriever for the knowledge base, combining vector search
        with BM25 keyword search for more robust retrieval.

        Args:
            k (int, optional): The number of documents to retrieve. Defaults to 5.

        Returns:
            EnsembleRetriever or VectorStoreRetriever: A hybrid retriever if splits are available,
                                                       otherwise a standard vector retriever.
        """
        vector_retriever = super().get_retriever(k)
        if self.splits:
            # Create a BM25 retriever for keyword-based search
            bm25_retriever = BM25Retriever.from_documents(self.splits, k=k)
            # Combine vector and BM25 retrievers using an ensemble retriever
            return EnsembleRetriever(
                retrievers=[vector_retriever, bm25_retriever],
                weights=[0.5, 0.5]  # Equal weighting for both retrievers
            )
        else:
            print("No document splits available for BM25. Using vector-only retrieval.")
            return vector_retriever
