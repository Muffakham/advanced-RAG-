from typing import List
from langchain_core.documents import Document
from langchain.retrievers.ensemble import EnsembleRetriever
from knowledgBaseManager import KnowledgeBaseManager
from memoryMangaer import MemoryManager
from graphState import GraphState

class Retrieval:
    """
    Retrieves and fuses documents from multiple vector stores.
    This class orchestrates the retrieval process by combining and ranking documents
    from both the knowledge base and the long-term conversation memory using an
    ensemble retriever and reciprocal rank fusion.
    """
    
    def __init__(self, kb_manager: KnowledgeBaseManager, mem_manager: MemoryManager):
        """
        Initializes the Retrieval class with knowledge base and memory managers.

        Args:
            kb_manager (KnowledgeBaseManager): The manager for the knowledge base.
            mem_manager (MemoryManager): The manager for the long-term memory.
        """
        # Create a hybrid retriever for the knowledge base
        kb_hybrid_retriever = kb_manager.get_retriever(k=5)
        
        # Create a retriever for the long-term memory
        mem_retriever = mem_manager.get_retriever(k=5)

        # Combine both into a single ensemble retriever for the entire retrieval process
        self.retriever = EnsembleRetriever(
            retrievers=[kb_hybrid_retriever, mem_retriever],
            weights=[0.7, 0.3]  # Prioritize knowledge base results
        )
    
    def _reciprocal_rank_fusion(self, results: List[List[Document]], k: int = 60) -> List[Document]:
        """
        Performs reciprocal rank fusion on a list of document lists.
        This method combines search results from multiple queries into a single
        re-ranked list.

        Args:
            results (List[List[Document]]): A list of lists, where each inner list contains
                                           documents retrieved for a specific sub-query.
            k (int, optional): A constant to prevent a very low rank from skewing the score.
                               Defaults to 60.

        Returns:
            List[Document]: A single, sorted list of unique documents based on their RRF score.
        """
        fused_scores = {}
        
        for result_list in results:
            for rank, doc in enumerate(result_list):
                content = doc.page_content  # Using page_content for uniqueness
                if content not in fused_scores:
                    fused_scores[content] = {"doc": doc, "score": 0}
                fused_scores[content]["score"] += 1.0 / (k + rank)
        
        # Sort documents by their total RRF score in descending order
        sorted_docs = sorted(fused_scores.values(), key=lambda x: x['score'], reverse=True)
        
        return [item['doc'] for item in sorted_docs]

    def retrieve(self, state: GraphState) -> dict:
        """
        Retrieves documents for each sub-query and fuses the results.

        Args:
            state (GraphState): The current state of the graph, containing sub-queries.

        Returns:
            dict: A dictionary containing the retrieved documents and the original question.
        """
        print("---RETRIEVING INFORMATION---")
        sub_queries = state.get('sub_queries', [state['question']])
        
        all_retrieved_docs = []
        
        for query in sub_queries:
            # Use the ensemble retriever to get documents from both knowledge base and memory
            docs = self.retriever.invoke(query)
            all_retrieved_docs.append(docs)
            print(f"Retrieved {len(docs)} documents for query: '{query}'")
        
        # Perform reciprocal rank fusion on the results of all sub-queries
        fused_docs = self._reciprocal_rank_fusion(all_retrieved_docs)

        # Get the top 5 documents from the fused list
        top_5_fused_docs = fused_docs[:5]
        
        print(f"Total unique documents retrieved and fused: {len(top_5_fused_docs)}")
        return {"retrieved_documents": top_5_fused_docs, "question": state['question']}
