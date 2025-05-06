# Banglalink Retailer App Chatbot

This is a RAG (Retrieval-Augmented Generation) based chatbot designed to assist users of the Banglalink Retailer App. The chatbot is built using FastAPI and LangGraph, and it responds to questions in the language of the query (English or Bangla). It strictly answers questions related to the Banglalink Retailer App.

## Features
- Provides step-by-step guidance for questions related to the Banglalink Retailer App.
- Responds in English for English questions and in Bangla for Bangla questions.
- Uses FastAPI to handle API calls through the `/ask` endpoint.
- Utilizes LangGraph and a RAG pipeline to retrieve information from documents and generate responses.
- Ignores irrelevant questions (e.g., "What is AI?") and only answers questions related to the Retailer App.
- Includes console logging at every step for debugging.

## Requirements
- **Python 3.8+**
- **Ollama Server** (uses `llama3` model as the LLM)
- **Dependencies**:
  - `fastapi`
  - `uvicorn`
  - `pydantic`
  - `langchain`
  - `langchain-chroma`
  - `langchain-huggingface`
  - `langchain-ollama`
  - `langchain-openai`
  - `langgraph`

## Setup Instructions

### 1. Clone the Project
```bash
git clone <your-repo-url>
cd rag-code-new
```

### 2. Create and Activate a Virtual Environment
```bash
python -m venv rag_env
.\rag_env\Scripts\activate  # On Windows
# or
source rag_env/bin/activate  # On Linux/Mac
```

### 3. Install Dependencies
```bash
pip install fastapi uvicorn pydantic langchain langchain-chroma langchain-huggingface langchain-ollama langchain-openai langgraph
```

### 4. Start the Ollama Server
- Install Ollama and start the server:
  ```bash
  ollama serve
  ```
- Download the `llama3` model:
  ```bash
  ollama pull llama3
  ```

### 5. Ingest Documents
- Ingest data from your Banglalink Retailer App documents (PDF, text, etc.) using the following script:
  ```python
  from langchain_chroma import Chroma
  from langchain_huggingface import HuggingFaceEmbeddings
  from langchain.text_splitter import RecursiveCharacterTextSplitter
  from langchain.document_loaders import PyPDFLoader

  # Load documents
  loader = PyPDFLoader("path_to_your_manual.pdf")
  documents = loader.load()

  # Split documents
  text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
  splits = text_splitter.split_documents(documents)

  # Create vector store
  embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
  vectorstore = Chroma.from_documents(documents=splits, embedding=embedding_model, persist_directory="./chroma_db")
  ```
- Replace the document path with the path to your manual file and run the script.

### 6. Start the FastAPI Server
```bash
uvicorn app:app --reload
```
- This will start the server at `http://127.0.0.1:8000`.

## Usage Instructions

### API Endpoint
- **URL**: `http://127.0.0.1:8000/ask`
- **Method**: POST
- **Body** (JSON):
  ```json
  {
      "question": "Write your question here"
  }
  ```
- **Examples**:
  - English question:
    ```json
    {
        "question": "How do I register a device in the Banglalink Retailer App?"
    }
    ```
  - Bangla question:
    ```json
    {
        "question": "বাংলালিংক রিটেইলার অ্যাপে ডিভাইস কীভাবে নিবন্ধন করব?"
    }
    ```

### Test with Postman
1. Open Postman.
2. Select **Method**: POST.
3. Enter **URL**: `http://127.0.0.1:8000/ask`.
4. Set **Body**: Raw -> JSON and enter the JSON above.
5. Click **Send**.
6. View the response.

### Test with Swagger UI
- Open `http://127.0.0.1:8000/docs` in your browser.
- Use the `/ask` endpoint to send questions.

## Changing the LLM
You can easily switch the LLM:
- **Use a different Ollama model**:
  ```python
  llm_config = LLMConfig(provider="ollama", model="mistral")
  ```
- **Use Open AI API**:
  ```python
  llm_config = LLMConfig(provider="openai", model="gpt-3.5-turbo", api_key="your-openai-api-key")
  ```
- Modify the `llm_config` line in `app.py` and restart the server.

## Troubleshooting
- **If Bangla responses are not working**:
  - Ensure that `./chroma_db` contains ingested data.
  - Check if your documents include Bangla content.
- **If you encounter errors**:
  - Check the console logs for error details.
  - Ensure all dependencies are installed.

## Contributing
To contribute to this project:
1. Fork the repository.
2. Make your changes and submit a pull request.

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.