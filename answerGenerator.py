from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from graphState import GraphState

class AnswerGenerator:
    """
    Generates the final answer based on retrieved documents and chat history.
    This class takes the retrieved context and the current conversation state
    to generate a concise and accurate response.
    """

    def __init__(self, llm: ChatOpenAI):
        """
        Initializes the AnswerGenerator with a language model and a prompt template.

        Args:
            llm (ChatOpenAI): The language model to be used for generating answers.
        """
        self.llm = llm
        # Prompt template for generating the final answer
        self.prompt_template = ChatPromptTemplate.from_messages(
            [
                ("system", "You are a helpful and truthful assistant. Answer the question using only the provided context. If the information is not in the context, truthfully say you don't have that information. Answer concisely."),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "Context: {context}\n\nQuestion: {question}"),
            ]
        )
        # Chain the prompt template with the language model
        self.chain = self.prompt_template | self.llm
    
    def generate(self, state: GraphState) -> dict:
        """
        Generates the final answer by invoking the language model with the
        retrieved context and chat history.

        Args:
            state (GraphState): The current state of the graph, containing the
                                question, retrieved documents, and chat history.

        Returns:
            dict: A dictionary containing the updated chat history with the
                  AI's response.
        """
        print("---GENERATING ANSWER---")
        chat_history = state.get('chat_history', [])
        question = state['question']
        retrieved_docs = state['retrieved_documents']
        
        # Combine the content of retrieved documents into a single context string
        context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
        
        # Invoke the chain with the necessary inputs
        response = self.chain.invoke(
            {
                "chat_history": chat_history,
                "context": context_text,
                "question": question
            }
        )
        
        # Create an AIMessage with the generated response content
        ai_response_message = AIMessage(content=response.content)
        
        # Append the user's question and the AI's response to the chat history
        return {"chat_history": chat_history + [HumanMessage(content=question), ai_response_message]}
