
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
    from both the knowledge base and the long-term conversation memory.
    """
    
    def __init__(self, kb_manager: KnowledgeBaseManager, mem_manager: MemoryManager):
        # Create a hybrid retriever for the knowledge base
        kb_hybrid_retriever = kb_manager.get_retriever(k=5)
        
        # Create a retriever for the long-term memory
        mem_retriever = mem_manager.get_retriever(k=5)

        # Combine both into a single ensemble retriever for the entire retrieval process
        self.retriever = EnsembleRetriever(
            retrievers=[kb_hybrid_retriever, mem_retriever],
            weights=[0.7, 0.3] # Adjust weights as needed
        )
    
    def _reciprocal_rank_fusion(self, results: List[List[Document]], k=60) -> List[Document]:
        """
        Performs reciprocal rank fusion on a list of document lists.
        
        Args:
            results: A list of lists, where each inner list contains documents
                     retrieved for a specific sub-query.
            k: A constant to prevent a very low rank from skewing the score.
            
        Returns:
            A single, sorted list of unique documents based on their RRF score.
        """
        fused_scores = {}
        
        for result_list in results:
            for rank, doc in enumerate(result_list):
                content = doc.page_content # replace it with page_id
                if content not in fused_scores:
                    fused_scores[content] = {"doc": doc, "score": 0}
                fused_scores[content]["score"] += 1.0 / (k + rank)
        
        # Sort documents by their total RRF score in descending order
        sorted_docs = sorted(fused_scores.values(), key=lambda x: x['score'], reverse=True)
        
        return [item['doc'] for item in sorted_docs]

    def retrieve(self, state: GraphState):
        print("---RETRIEVING INFORMATION---")
        sub_queries = state.get('sub_queries', [state['question']])
        
        all_retrieved_docs = []
        
        for query in sub_queries:
            # Use the single ensemble retriever to get all relevant documents from both sources
            docs = self.retriever.invoke(query)
            all_retrieved_docs.append(docs)
            print(f"Retrieved {len(docs)} documents for query: '{query}'")
        
        # Perform reciprocal rank fusion on the results of all sub-queries
        fused_docs = self._reciprocal_rank_fusion(all_retrieved_docs)

        # Get the top 5 documents from the fused list
        top_5_fused_docs = fused_docs[:5]
        
        print(f"Total unique documents retrieved and fused: {len(top_5_fused_docs)}")
        return {"retrieved_documents": top_5_fused_docs, "question": state['question']}


if __name__ == "__main__":
    try:
        # Initialize LLM and Embeddings

        from langchain_openai import ChatOpenAI, OpenAIEmbeddings
        from queryDeconstructor import QueryDeconstructor

        llm = ChatOpenAI(temperature=0.0)
        embeddings = OpenAIEmbeddings()

        # Initialize Knowledge Base and Memory Managers
        kb_manager = KnowledgeBaseManager(
            embeddings = embeddings,
            kb_path="kb",
            persist_directory="./vector_stores/knowledge_base",
            force_reindex=True
        )
        
        mem_manager = MemoryManager(
            embeddings=embeddings,
            persist_directory="./vector_stores/long_term_memory"
        )
        
        # Initialize Deconstructor and Retrieval components
        deconstructor = QueryDeconstructor(llm)
        retrieval = Retrieval(kb_manager, mem_manager)

        # Create a sample query
        initial_state = GraphState()
        initial_state['question']  = "What are the common applications of RAGs and how is memory used in this system?"

        # Step 1: Deconstruct the query
        deconstructed_state = deconstructor.deconstruct(initial_state)

        # Step 2: Retrieve documents based on sub-queries
        final_state = retrieval.retrieve(deconstructed_state)

        # Print the final result to verify
        print("\n--- Final Result ---")
        print("Retrieved Documents:", final_state.get("retrieved_documents", []))
        print("Original Question:", final_state.get("question", ""))
        
    except Exception as e:
        print(f"An error occurred: {e}")
        print("Please ensure you have configured your LLM API key and a 'kb' directory with .txt files correctly.")
