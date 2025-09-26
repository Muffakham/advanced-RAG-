
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from graphState import GraphState


class AnswerGenerator:
    """Generates the final answer based on retrieved documents and chat history."""

    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
        self.prompt_template = ChatPromptTemplate.from_messages(
            [
                ("system", "You are a helpful and truthful assistant. Answer the question using only the provided context. If the information is not in the context, truthfully say you don't have that information. Answer concisely."),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "Context: {context}\n\nQuestion: {question}"),
            ]
        )
        self.chain = self.prompt_template | self.llm
    
    def generate(self, state: GraphState):
        print("---GENERATING ANSWER---")
        chat_history = state.get('chat_history', [])
        question = state['question']
        retrieved_docs = state['retrieved_documents']
        context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
        
        response = self.chain.invoke(
            {
                "chat_history": chat_history,
                "context": context_text,
                "question": question
            }
        )
        
        ai_response_message = AIMessage(content=response.content)
        return {"chat_history": chat_history + [HumanMessage(content=question), ai_response_message]}

if __name__ == "__main__":
    try:
        # Initialize LLM and Embeddings

        from langchain_openai import ChatOpenAI, OpenAIEmbeddings
        from queryDeconstructor import QueryDeconstructor
        from retrievalManager import Retrieval
        from knowledgBaseManager import KnowledgeBaseManager
        from memoryMangaer import MemoryManager

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

        ans = AnswerGenerator(llm)
        resp = ans.generate(final_state)



        # Print the final result to verify
        print("\n--- Final Result ---")
        print("Retrieved Documents:", final_state.get("retrieved_documents", []))
        print("Original Question:", final_state.get("question", ""))
        print("final response: ", resp)
        
    except Exception as e:
        print(f"An error occurred: {e}")
        print("Please ensure you have configured your LLM API key and a 'kb' directory with .txt files correctly.")
