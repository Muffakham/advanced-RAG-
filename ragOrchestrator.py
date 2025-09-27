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

# Load environment variables from .env file
load_dotenv()

class RAGChatbot:
    """
    A RAG chatbot that uses LangGraph to manage conversational state and memory.
    It orchestrates various sub-components to handle user queries,
    including knowledge base management, memory management, query deconstruction,
    information retrieval, and answer generation.
    """
    def __init__(self, debug: bool = False):
        """
        Initializes the RAGChatbot with necessary components and configurations.

        Args:
            debug (bool): If True, enables streaming of debug output for each step
                          in the LangGraph workflow. Defaults to False.
        """
        print("Initializing RAGChatbot orchestrator...")
        self.debug = debug
        
        # Initialize shared components for embeddings and language model
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
        """
        Defines and compiles the LangGraph workflow.
        The workflow consists of several nodes representing different stages
        of the RAG process: storing memory, deconstructing the query,
        retrieving information, and generating an answer.

        Returns:
            Compiled LangGraph application.
        """
        workflow = StateGraph(GraphState)
        
        # Add nodes to the workflow
        workflow.add_node("store_memory", self.mem_manager.store)
        workflow.add_node("deconstruct_query", self.query_deconstructor.deconstruct)
        workflow.add_node("retrieve_info", self.retrieval.retrieve)
        workflow.add_node("generate", self.answer_generator.generate)
        
        # Define the edges (flow) between the nodes
        workflow.add_edge(START, "store_memory")
        workflow.add_edge("store_memory", "deconstruct_query")
        workflow.add_edge("deconstruct_query", "retrieve_info")
        workflow.add_edge("retrieve_info", "generate")
        workflow.add_edge("generate", END)
        
        # Compile the workflow with a memory-based checkpointer for state persistence
        return workflow.compile(checkpointer=MemorySaver())

    def run_chat_loop(self):
        """
        Main chat loop for user interaction.
        Continuously prompts the user for input, processes the query through
        the RAG workflow, and prints the AI's response.
        Supports debug mode for streaming intermediate steps.
        """
        print("--- LangGraph RAG Chatbot with Hybrid Memory ---")
        print("Ask a question (type 'exit' to quit).")
        
        # Define a thread ID for conversational memory
        thread_id = "rag-chat-1"
        
        while True:
            user_input = input("\nHuman: ")
            if user_input.lower() == 'exit':
                print("--- Goodbye! ---")
                break

            # Configuration for the LangGraph run, including the thread ID
            config = {"configurable": {"thread_id": thread_id}}
            
            try:
                # Initialize inputs for the graph
                inputs = {"question": user_input, "chat_history": []}
                
                # Attempt to retrieve existing chat history for the thread
                try:
                    state = self.app.get_state(config)
                    if state and state.values.get('chat_history'):
                        inputs['chat_history'] = state.values['chat_history']
                except Exception:
                    # If no state or chat history is found, proceed with empty history
                    pass

                # Run the graph based on debug mode
                if self.debug:
                    print("--- Streaming debug output ---")
                    # Stream outputs of each step in debug mode
                    for s in self.app.stream(inputs, config=config):
                        print(s)
                        print("----")
                    print("--- End of stream ---")
                else:
                    # Invoke the graph to get the final state directly
                    self.app.invoke(inputs, config=config)
                
                # Retrieve the final state and extract the last message
                final_state = self.app.get_state(config)
                if final_state and final_state.values.get('chat_history'):
                    last_message = final_state.values['chat_history'][-1]
                    print(f"\nAI: {last_message.content}")

            except Exception as e:
                print(f"An error occurred: {e}")
                print("Please try again or restart the application.")

if __name__ == "__main__":
    # Instantiate and run the chatbot in debug mode
    chatbot = RAGChatbot(debug=True)
    chatbot.run_chat_loop()
