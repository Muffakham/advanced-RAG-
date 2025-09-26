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
    """Manages the creation and access of the knowledge base vector store."""
    
    def __init__(self, 
                 embeddings: OpenAIEmbeddings = None,
                 kb_path: str = "kb",
                 persist_directory: str = "./vector_stores/knowledge_base",
                 collection_name: str = "rag_knowledge",
                 chunk_size: int = 1000,
                 chunk_overlap: int = 200,
                 force_reindex: bool = False):
        
        self.embeddings = embeddings or OpenAIEmbeddings()
        self.kb_path = Path(kb_path)
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.force_reindex = force_reindex
        

        docs = self._load_documents_from_kb()
        self.splits = self._split_documents(docs)
        # Call the parent class to handle vector store creation
        super().__init__(
            embeddings=self.embeddings,
            persist_directory=persist_directory,
            collection_name=collection_name,
            force_reindex=force_reindex,
            splits=self.splits
        )
    
    def _load_documents_from_kb(self):
        """Load all .txt files from kb_path directory."""
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

    
    def _split_documents(self, docs: List[Document]):
        """Split documents into chunks using RecursiveCharacterTextSplitter."""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size, 
            chunk_overlap=self.chunk_overlap
        )
        return text_splitter.split_documents(docs) if docs else []


    def get_retriever(self, k: int = 5):
        """
        Creates a hybrid retriever for the knowledge base.
        """
        vector_retriever = super().get_retriever(k)
        if self.splits:
            bm25_retriever = BM25Retriever.from_documents(self.splits, k=k)
            return EnsembleRetriever(
                retrievers=[vector_retriever, bm25_retriever],
                weights=[0.5, 0.5]
            )
        else:
            print("No document splits available for BM25. Using vector-only retrieval.")
            return vector_retriever


if __name__ == "__main__":
    # Example usage:
    # This will create a new one from the 'kb' directory.
    print("--- Initializing KnowledgeBaseManager ---")
    kb_manager = KnowledgeBaseManager(
        kb_path="kb",
        persist_directory="./my_vector_stores/knowledge_base",
        collection_name="rag_knowledge_base",
        chunk_size=200,
        chunk_overlap=50,
        force_reindex=False
    )

    # Get the hybrid retriever
    retriever = kb_manager.get_retriever(k=3)
    
    # Perform a test retrieval
    test_query = "What are the common applications of LLMs?"
    print(f"\n--- Testing retrieval for query: '{test_query}' ---")
    retrieved_docs = retriever.invoke(test_query)
    
    # Print the retrieved documents
    print(f"Retrieved {len(retrieved_docs)} documents:")
    for i, doc in enumerate(retrieved_docs):
        print(f"\n--- Document {i+1} ---")
        print(doc.page_content)
        print("-" * 20)



    #this will load an existing vector store

    print("test the reasigning ------------------  ")
    del kb_manager
    kb_manager = KnowledgeBaseManager(
        kb_path="kb",
        persist_directory="./my_vector_stores/knowledge_base",
        collection_name="rag_knowledge_base",
        chunk_size=200,
        chunk_overlap=50,
        force_reindex=False
    )

    retriever = kb_manager.get_retriever(k=3)
    
    # Perform a test retrieval
    test_query = "What are the common applications of RAG?"
    print(f"\n--- Testing retrieval for query: '{test_query}' ---")
    retrieved_docs = retriever.invoke(test_query)
    
    # Print the retrieved documents
    print(f"Retrieved {len(retrieved_docs)} documents:")
    for i, doc in enumerate(retrieved_docs):
        print(f"\n--- Document {i+1} ---")
        print(doc.page_content)
        print("-" * 20)

