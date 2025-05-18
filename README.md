# Project: Advanced Question Answering System for Academic Regulations

## 1. Project Overview

This project implements a sophisticated Question Answering (QA) system designed to interact with a PDF document containing "General Academic Regulations." It leverages advanced Natural Language Processing (NLP) techniques, including Retrieval-Augmented Generation (RAG), various text embedding models, vector databases for efficient similarity search, and LangChain for orchestrating LLM interactions and agent-based web searches. The system aims to provide accurate answers based on the document's content and can be extended with external knowledge. A notable feature is its capability to process queries and deliver responses in Georgian, demonstrating multilingual adaptability.

## 2. Core Workflow & Implementation Steps

The project unfolds in a series of well-defined stages, from data ingestion to intelligent response generation:

### Step 1: Environment Setup and Initialization

-   **API Key Management**: Securely loads the Google API key using `python-dotenv` from an external `.env` file. This is crucial for accessing Google's Generative AI models.
-   **LLM Initialization**: The `ChatGoogleGenerativeAI` model (specifically `gemini-2.0-flash`) is initialized. This Large Language Model (LLM) serves as the core reasoning and generation engine.

    ```python
    # from HW2.ipynb (cell_id: 4ef6ca15)
    # initializing model
    model_name = 'gemini-2.0-flash'
    llm_model = ChatGoogleGenerativeAI(
        model=model_name,
        google_api_key=api_key,
        temperature=0.0,
        convert_system_message_to_human=True
    )
    print(f"Initialized Gemini Model: {model_name}")
    ```

### Step 2: Document Loading and Preprocessing

-   **PDF Ingestion**: The primary data source, `General_Academic_Regulations_21.02.2025-eng.pdf`, is loaded using `PyPDFLoader` from LangChain.
-   **Text Splitting**: To handle the large document context and prepare it for embedding, the text is split into smaller, manageable chunks. `RecursiveCharacterTextSplitter` is employed for its ability to semantically divide text based on common separators, maintaining context.

    ```python
    # from HW2.ipynb (cell_id: 9b1e363d, 982b396a)
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    r_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=70
    )
    docs = loader.load() # Assuming loader is PyPDFLoader instance
    doc_split = r_splitter.split_documents(docs)
    ```

### Step 3: Text Embedding and Vector Store Creation

Two different embedding strategies were explored:

1.  **BERT-based Embeddings (Initial Experiment)**:
    *   The `bert-base-uncased` model from Hugging Face Transformers was used to generate initial embeddings for the document chunks.
    *   A custom `get_embedding` function tokenized text and passed it through the BERT model to extract sentence/chunk embeddings.

2.  **SentenceTransformer Embeddings (Primary Approach)**:
    *   The `sentence-transformers/all-MiniLM-L6-v2` model was chosen for its efficiency and effectiveness in generating high-quality sentence embeddings.
    *   Embeddings were generated for each document chunk.

        ```python
        # from HW2.ipynb (cell_id: b9b75844, 550f830a)
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        document_embeddings_new = [model.encode(chunk.page_content) for chunk in doc_split]
        ```

-   **FAISS Vector Store**: Facebook AI Similarity Search (FAISS) was used to create an indexed vector store for the document embeddings (primarily the `all-MiniLM-L6-v2` embeddings). This allows for highly efficient similarity searches, which is the backbone of the retrieval mechanism in RAG.

    ```python
    # from HW2.ipynb (cell_id: 08da7b4a)
    import faiss
    import numpy as np

    document_embeddings_new_np = np.array(document_embeddings_new).astype('float32')
    dimension_of_embeddings_new = document_embeddings_new_np.shape[1]
    index = faiss.IndexFlatL2(dimension_of_embeddings_new)
    index.add(document_embeddings_new_np)
    faiss.write_index(index, 'document_embeddings.index') # Optional: save index
    ```

### Step 4: Retrieval Mechanism

-   When a user query is received, it's first converted into an embedding using the same `all-MiniLM-L6-v2` model.
-   This query embedding is then used to search the FAISS index, retrieving the `k` most semantically similar document chunks. These chunks form the "context" for the LLM.

    ```python
    # from HW2.ipynb (cell_id: 156a471e, 75222a98)
    query_text = "3.1 GPA" # Example query
    query_embedding = model.encode(query_text)
    k = 10
    distances, indices = index.search(np.array([query_embedding]), k)
    retrieved_chunks = [doc_split[i].page_content for i in indices[0]]
    context = "\\n\\n".join(retrieved_chunks)
    ```

### Step 5: Generation and Augmentation (RAG Core & Beyond)

-   **LLM Chains for Sophisticated Processing**: A `SimpleSequentialChain` from LangChain was constructed to process the query and context, and to generate a final, polished answer. This chain involved multiple sub-chains:
    1.  `chain_inference`: An initial chain to interpret the user's question and deduce relevant information from the provided context (though the prompt in the notebook also includes a hardcoded Georgian phrase, suggesting a complex instruction).
    2.  `chain_clean`: A subsequent chain intended to refine or add information to the output of the first chain.
    3.  `chain_predict`: A final chain tasked with translating the processed information into sophisticated Georgian.

    ```python
    # from HW2.ipynb (cell_id: e1f14bed)
    from langchain.chains import LLMChain, SimpleSequentialChain
    from langchain.prompts import PromptTemplate

    # Example prompt for the final translation chain
    topic_prompt4 = PromptTemplate(
        input_variables=['text'],
        template='Write the following in Georgian, making it sophisticated: {text}'
    )
    chain_predict = LLMChain(llm=llm_model, prompt=topic_prompt4)

    # Assuming chain_inference and chain_clean are defined similarly
    # main_chain = SimpleSequentialChain(chains=[chain_inference, chain_clean, chain_predict])
    # output = main_chain.run(docs) # Input 'docs' seems to be a placeholder; typically it would be the query or context
    ```
    *Self-correction during thought process: The `main_chain.run(docs)` in the notebook is a bit unusual as input. Typically, a sequential chain would process the output of one chain as input to the next, starting with an initial input like the user's query or the retrieved context. The notebook's implementation appears to focus on processing/translating a more general task described in the initial prompt.*

-   **Agent-based Web Search**: For queries that might require information beyond the scope of the provided PDF, a LangChain agent was implemented.
    *   This agent utilizes the `DuckDuckGoSearchRun` tool to perform live web searches.
    *   It employs a "zero-shot-react-description" agent type, allowing the LLM to reason about which tool to use (in this case, web search) and how to use it based on the query.

    ```python
    # from HW2.ipynb (cell_id: 606f8ac6)
    from langchain.agents import initialize_agent, Tool
    from langchain.tools import DuckDuckGoSearchRun

    search_tool = DuckDuckGoSearchRun()
    tools = [
        Tool(
            name="DuckDuckGoSearch",
            func=search_tool.run,
            description="Search the web for up-to-date information"
        )
    ]
    agent = initialize_agent(
        tools,
        llm_model,
        agent="zero-shot-react-description",
        verbose=True
    )
    # response = agent.run("now search information what evaluation system is there other than GPA, and retrieve scoring system")
    ```

## 3. Key Technologies Leveraged

-   **Python 3**: Core programming language.
-   **LangChain**: Framework for developing applications powered by language models. Used for:
    -   Document Loaders (`PyPDFLoader`)
    -   Text Splitters (`RecursiveCharacterTextSplitter`)
    -   LLM Wrappers (`ChatGoogleGenerativeAI`)
    -   Prompt Engineering (`PromptTemplate`, `ChatPromptTemplate`)
    -   Chains (`LLMChain`, `SimpleSequentialChain`)
    -   Agents (`initialize_agent`)
    -   Tools (`DuckDuckGoSearchRun`)
-   **Google Generative AI**: Provider of the `gemini-2.0-flash` LLM.
-   **Hugging Face Transformers**: For accessing pre-trained models like BERT (`bert-base-uncased`).
-   **SentenceTransformers**: For high-quality sentence and text embeddings (`all-MiniLM-L6-v2`).
-   **FAISS**: For efficient similarity search in high-dimensional vector spaces.
-   **Pandas**: (Imported, though not heavily used in the final workflow shown for QA) For data manipulation.
-   **NumPy**: For numerical operations, especially with embeddings.
-   **python-dotenv**: For managing environment variables (API keys).
-   **Pydantic**: (Imported) For data validation, often used with LangChain's output parsers.

## 4. How to Run

1.  **Clone the Repository/Setup Project**: Ensure you have the `HW2.ipynb` notebook and the `General_Academic_Regulations_21.02.2025-eng.pdf` file.
2.  **Create Environment File**: Create a `.env` file in the project root with your Google API Key:
    ```
    API_KEY="YOUR_GOOGLE_API_KEY"
    ```
3.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
    (Ensure `requirements.txt` is generated as described at the beginning).
4.  **Execute the Jupyter Notebook**: Run the cells in `HW2.ipynb` sequentially. Pay attention to file paths if they are hardcoded.

## 5. Conclusion

This project successfully demonstrates the construction of an advanced RAG-based QA system. By integrating robust embedding techniques, efficient vector search with FAISS, and the powerful reasoning capabilities of Google's Gemini LLM orchestrated via LangChain, the system can effectively retrieve relevant information from academic regulations and provide structured, potentially multilingual, responses. The inclusion of an agent for external web searches further enhances its utility by allowing it to access information beyond the confines of the source document.
