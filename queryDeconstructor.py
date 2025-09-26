from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from graphState import GraphState


class QueryDeconstructor:
    """Breaks down a complex query into sub-queries using an LLM."""
    
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
        self.prompt_template = ChatPromptTemplate.from_messages(
            [
                ("system", "You are an expert query deconstructor. Given a user's question, break it down into atomic, self-contained sub-queries for information retrieval. Respond with a comma-separated string of the sub-queries, with no other text. example: sub_query1, sub_query2, ... sub_queryN"),
                ("human", "Question: {question}"),
            ]
        )
        self.chain = self.prompt_template | self.llm
        
    def deconstruct(self, state: GraphState):
        print("---DECONSTRUCTING QUERY---")
        question = state['question']
        response = self.chain.invoke({"question": question})
        sub_queries = [q.strip() for q in response.content.split(',')]
        print(f"Deconstructed into: {sub_queries}")
        return {"sub_queries": sub_queries, "question": question}



if __name__ == "__main__":
    #simple test for the module
    # Ensure you have your OpenAI API key set up in your environment
    try:
        llm = ChatOpenAI(temperature=0.0)

        # Create a sample query
        state = {"question": "What is a hybrid retrieval system and how is memory used in this system?"}

        # Initialize the deconstructor
        deconstructor = QueryDeconstructor(llm)

        # Deconstruct the query
        result = deconstructor.deconstruct(state)

        # Print the final result to verify
        print("\n--- Final Result ---")
        print(result)

    except Exception as e:
        print(f"An error occurred: {e}")
        print("Please ensure you have configured your LLM API key correctly.")
