from knowledgBaseManager import KnowledgeBaseManager
from memoryMangaer import MemoryManager
from retrievalManager import Retrieval
from answerGenerator import AnswerGenerator
from queryDeconstructor import QueryDeconstructor
from graphState import GraphState
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.memory import MemorySaver
from dotenv import load_dotenv
import os

load_dotenv()

class RAGChatbot:
    """
    A RAG chatbot that uses LangGraph to manage conversational state and memory.
    It orchestrates the various sub-components to handle user queries.
    """
    def __init__(self, debug: bool = False):
        print("Initializing RAGChatbot orchestrator...")
        self.debug = debug
        
        # Initialize shared components
        self.embeddings = OpenAIEmbeddings()
        self.llm = ChatOpenAI(model="gpt-4o", temperature=0)

        # Initialize the component managers
        self.kb_manager = KnowledgeBaseManager(self.embeddings)
        self.mem_manager = MemoryManager(self.embeddings)
        self.query_deconstructor = QueryDeconstructor(self.llm)
        self.retrieval = Retrieval(self.kb_manager, self.mem_manager)
        self.answer_generator = AnswerGenerator(self.llm)

        # Compile the LangGraph workflow
        self.app = self._compile_graph()
        print("Chatbot ready to go!")

    def _compile_graph(self):
        """Defines and compiles the LangGraph workflow."""
        workflow = StateGraph(GraphState)
        
        workflow.add_node("store_memory", self.mem_manager.store)
        workflow.add_node("deconstruct_query", self.query_deconstructor.deconstruct)
        workflow.add_node("retrieve_info", self.retrieval.retrieve)
        workflow.add_node("generate", self.answer_generator.generate)
        
        workflow.add_edge(START, "store_memory")
        workflow.add_edge("store_memory", "deconstruct_query")
        workflow.add_edge("deconstruct_query", "retrieve_info")
        workflow.add_edge("retrieve_info", "generate")
        workflow.add_edge("generate", END)
        
        return workflow.compile(checkpointer=MemorySaver())

    def run_chat_loop(self):
        """Main chat loop for user interaction."""
        print("--- LangGraph RAG Chatbot with Hybrid Memory ---")
        print("Ask a question (type 'exit' to quit).")
        
        thread_id = "rag-chat-1"
        
        while True:
            user_input = input("\nHuman: ")
            if user_input.lower() == 'exit':
                print("--- Goodbye! ---")
                break

            config = {"configurable": {"thread_id": thread_id}}
            
            try:
                inputs = {"question": user_input, "chat_history": []}
                
                try:
                    state = self.app.get_state(config)
                    if state and state.values.get('chat_history'):
                        inputs['chat_history'] = state.values['chat_history']
                except Exception:
                    pass

                if self.debug:
                    print("--- Streaming debug output ---")
                    for s in self.app.stream(inputs, config=config):
                        print(s)
                        print("----")
                    print("--- End of stream ---")
                else:
                    self.app.invoke(inputs, config=config)
                
                final_state = self.app.get_state(config)
                if final_state and final_state.values.get('chat_history'):
                    last_message = final_state.values['chat_history'][-1]
                    print(f"\nAI: {last_message.content}")

            except Exception as e:
                print(f"An error occurred: {e}")
                print("Please try again or restart the application.")

if __name__ == "__main__":
    chatbot = RAGChatbot(debug=True)
    chatbot.run_chat_loop()
