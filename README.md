# Advanced RAG Chatbot with LangGraph and Hybrid Memory

This project implements an advanced Retrieval-Augmented Generation (RAG) chatbot using LangChain, LangGraph, and a hybrid memory system. The chatbot is designed to answer questions based on a knowledge base while maintaining conversational context.

## Features

*   **Retrieval-Augmented Generation (RAG):** The chatbot retrieves relevant information from a knowledge base before generating a response, ensuring answers are grounded in provided data.
*   **LangGraph Orchestration:** The entire workflow is orchestrated using LangGraph, providing a clear and maintainable structure for the different stages of processing a query.
*   **Hybrid Memory:** The chatbot utilizes a hybrid memory system, combining both a traditional chat history and a vector-based memory for more nuanced context management.
*   **Long-Term Memory:** Utilizes a vector database to store and access previous messages, enabling the chatbot to maintain context over extended conversations.
*   **Query Deconstruction:** User queries are deconstructed into sub-queries to identify the core question and any necessary context from the chat history.
*   **Reciprocal Rank Fusion (RRF):** Employs RRF to effectively combine and re-rank documents obtained from multiple sub-queries, ensuring the most relevant information is presented.
*   **Debug Mode:** A debug flag allows for streaming the output of each step in the LangGraph workflow, providing insight into the chatbot's internal workings.
*   **Vector Store Management:** The project includes functionality for creating and managing a vector store for the knowledge base.

## Project Structure

```

├── answerGenerator.py
├── graphState.py
├── kb
├── knowledgBaseManager.py
├── memoryMangaer.py
├── queryDeconstructor.py
├── ragOrchestrator.py
├── requirements.txt
├── retrievalManager.py
├── setup.sh
├── vectorStoreManager.py
```

## Setup and Usage

1.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

2.  **Set up your environment:**
    *   Create a `.env` file in the root of the project.
    *   Add your OpenAI API key to the `.env` file:
        ```
        OPENAI_API_KEY="your-api-key"
        ```

3.  **Create the virtual environment:**
    *   Run the `setup.sh` script to create the virtual env for the project.
        ```bash
        ./setup.sh
        ```

4. **Activate the virtual env:**
    *   Run the following command:
        ```
        <your virtual env name>/bin/activate
        ```


5. **Run the chatbot:**
    ```bash
    python ragOrchestrator.py
    ```

    The chatbot will start, and you can begin asking questions. To see the debug output, you can run the chatbot with the `debug` flag set to `True` in `ragOrchestrator.py`.

## How it Works

The chatbot operates in the following sequence, orchestrated by LangGraph:

1.  **Store Memory:** The user's query is stored in the chat history.
2.  **Deconstruct Query:** The query is analyzed and broken down into sub-queries. A search query is generated based on the current question and the chat history.
3.  **Retrieve Info:** Relevant documents are retrieved from the knowledge base and the vector-based memory for each sub-query. Reciprocal Rank Fusion (RRF) is then applied to combine these documents into a single, highly relevant list.
4.  **Generate:** The retrieved information and the original query are passed to the language model to generate a final answer.

This entire process is managed as a stateful graph, allowing for robust and flexible conversational AI.
