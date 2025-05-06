# Banglalink Retailer App RAG Chatbot

## Project Structure

```
rag-code-new/
│
├── app/
│   ├── __init__.py
│   ├── api.py              # FastAPI endpoints (chatbot)
│   ├── main.py             # FastAPI app entrypoint
│   ├── rag_pipeline.py     # RAG pipeline logic
│   ├── llm_config.py       # LLM config and initialization
│   ├── utils.py            # Utility functions
│   └── manual_upload.py    # Manual upload & vector DB processing
│
├── chroma_db/              # Vector DB files
├── uploaded_manuals/       # Uploaded files
├── requirements.txt
├── README.MD
├── user_manual.txt
└── ...
```

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

## Setup & Usage

1. **Clone the Project**
   ```bash
   git clone <your-repo-url>
   cd rag-code-new
   ```

2. **Create and Activate a Virtual Environment**
   ```bash
   python -m venv rag_env
   .\rag_env\Scripts\activate  # On Windows
   # or
   source rag_env/bin/activate  # On Linux/Mac
   ```

3. **Install dependencies** (in your virtual environment):
   ```bash
   pip install -r requirements.txt
   ```

4. **Start the Ollama Server**
   - Install Ollama and start the server:
     ```bash
     ollama serve
     ```
   - Download the `llama3` model:
     ```bash
     ollama pull llama3
     ```

5. **Upload a manual:**
   - POST a file to `/upload_manual` (e.g., using Postman or curl)
   - Example (using curl):
     ```bash
     curl -F "file=@user_manual.txt" http://localhost:8000/upload_manual
     ```

6. **Run the FastAPI server:**
   ```bash
   uvicorn app.main:app --reload
   ```

7. **Ask a question:**
   - POST to `/ask` with JSON body `{ "question": "your question here" }`
   - Example (using curl):
     ```bash
     curl -X POST http://localhost:8000/ask -H "Content-Type: application/json" -d '{"question": "How do I register a device?"}'
     ```

## Notes
- All business logic is now modularized in the `app/` package.
- The old `app.py` and `process_manual.py` are deprecated.
- For custom logic or tests, add new modules under `app/` or a `tests/` folder.

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
- Modify the `llm_config` line in `app/llm_config.py` and restart the server.

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