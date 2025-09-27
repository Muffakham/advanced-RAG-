from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from graphState import GraphState

class QueryDeconstructor:
    """
    Breaks down a complex user query into smaller, atomic sub-queries using an LLM.
    This helps in retrieving more targeted and relevant information from the vector stores.
    """
    
    def __init__(self, llm: ChatOpenAI):
        """
        Initializes the QueryDeconstructor with a language model and a prompt template.

        Args:
            llm (ChatOpenAI): The language model to be used for deconstructing queries.
        """
        self.llm = llm
        # Prompt template to instruct the LLM on how to deconstruct the query
        self.prompt_template = ChatPromptTemplate.from_messages(
            [
                ("system", "You are an expert query deconstructor. Given a user's question, break it down into atomic, self-contained sub-queries for information retrieval. Respond with a comma-separated string of the sub-queries, with no other text. example: sub_query1, sub_query2, ... sub_queryN"),
                ("human", "Question: {question}"),
            ]
        )
        # Chain the prompt template with the language model
        self.chain = self.prompt_template | self.llm
        
    def deconstruct(self, state: GraphState) -> dict:
        """
        Deconstructs the user's question into a list of sub-queries.

        Args:
            state (GraphState): The current state of the graph, containing the user's question.

        Returns:
            dict: A dictionary containing the list of generated sub-queries and the original question.
        """
        print("---DECONSTRUCTING QUERY---")
        question = state['question']
        
        # Invoke the chain to get the deconstructed queries
        response = self.chain.invoke({"question": question})
        
        # Parse the comma-separated response into a list of sub-queries
        sub_queries = [q.strip() for q in response.content.split(',')]
        print(f"Deconstructed into: {sub_queries}")
        
        return {"sub_queries": sub_queries, "question": question}
