# Aaltoes AI Search

Aaltoes AI Search is a Retrieval-Augmented Generation (RAG) application designed to facilitate intelligent search and question-answering over documents related to Aaltoes (Aalto Entrepreneurship Society). The application leverages Google Drive as a document source, ChromaDB as a vector store, and integrates with state-of-the-art language models like OpenAI's GPT and Google's Gemini.

## Features

- **Document Retrieval**: Fetch documents from Google Drive based on year and file type.
- **Question-Answering**: Generate answers to user queries using retrieved documents.
- **Embedding Models**: Support for OpenAI and Google embedding models.
- **Language Models**: Integration with OpenAI GPT models and Google's Gemini models.
- **Document Parsing**: Extract and process content from various file types, using `unstructured` library.
- **Translation**: Translate Finnish documents to English for better accessibility.
- **Gradio Interface**: User-friendly web interface for querying and exploring results.
- **FastAPI Integration**: Expose indexing and retrieval functionalities via a FastAPI-based REST API.
- **Concurrency and Parallelism**: Leverage `asyncio` and `multiprocessing` to speed up document indexing and retrieval.

## TODO
- **Index files based on their type**. Parsing time depends on the type and length of a file.
- **Agentic coding for spreadsheets**

## Prerequisites

- Python 3.12 or higher.
- `uv` as the project manager.
- Google Cloud credentials for accessing Google Drive.
- API keys for OpenAI and Google Gemini.

## Installation

1. Clone the repository:
    ```bash
    git clone <repository-url>
    cd <repository-folder>
    ```

2. Set up your environment:
    - Create a `.env` file in the project root.
    - Add the following variables:
        ```env
        GEMINI_API_KEY=<your-google-gemini-api-key>
        OPENAI_API_KEY=<your-openai-api-key>
        CHROMADB_PERSISTED_PATH=./chroma_db
        ```

3. uv will create the venv and install dependencies for you when you run the app.

## Quick Start

### Using the CLI

1. **Index Documents**:
    To index documents from Google Drive, run:
    ```bash
    uv run main.py --mode index
    ```
    use --help to see other options

2. **Retrieve Documents**:
    To retrieve documents from Google Drive, run:
    ```bash
    uv run main.py --mode retrieve --query <your query>
    ```
    use --help to see other options

### Using Gradio
**Search and Retrieve**:
    Use the Gradio interface to input your query, select parameters like year, model, and embedding function, and retrieve answers along with document references. To launch the Gradio interface, run:
    ```bash
    uv run gr_app.py
    ```

### Using the FastAPI Endpoints

1. **Start the FastAPI Server**:
    Run the FastAPI app in dev mode:
    ```bash
    fastapi dev fapi_app.py
    ```
    for production mode:
    ```bash
    fastapi run fapi_app.py
    ```

2. **Index Documents**:
    Send a POST request to the `/index` endpoint with the required parameters. Example request body:
    ```json
    {
        "mode": "index",
        "model": "gpt-4o",
        "emb_func": "openai",
        "indx_years": [2020, 2021]
    }
    ```

3. **Retrieve Documents**:
    Send a POST request to the `/retrieve` endpoint with the required parameters. Example request body:
    ```json
    {
        "mode": "retrieve",
        "model": "gpt-4o",
        "emb_func": "openai",
        "query": "What was the purpose of Aaltoes?",
        "top_k": 5,
        "retr_year": "2021"
    }
    ```

    Example response:
    ```json
    {
        "response": "Aaltoes was focused on fostering entrepreneurship...",
        "reference_files": "- Document 1: \"Event Summary 2021\" from year 2021\n- Document 2: \"Partnerships 2021\" from year 2021"
    }
    ```

4. **Check API Status**:
    Access the root endpoint to verify the API is running:
    ```bash
    curl http://127.0.0.1:8000/
    ```

## Configuration

The application uses a `APIConfig` class (based on Pydantic's `BaseModel`) to validate and manage configuration parameters. Key parameters include:

- `mode`: Operation mode (`index` or `retrieve`).
- `model`: Language model to use (e.g., `gpt-4o`, `gemini-2.0`).
- `emb_func`: Embedding function (`openai` or `google`).
- `indx_years`: List of years to index or `"Full"` for all years.
- `query`: Query string for retrieval.
- `top_k`: Number of top results to retrieve.
- `retr_year`: Year to filter retrieval results or `"Full"` for all years.
- `file_type`: File type to filter (e.g., `PDF`, `MSWords`, or `"Full"` for all types).

## Notes

- The FastAPI app provides a programmatic interface for indexing and retrieval, making it easier to integrate with other systems.
- The CLI remains available for quick local testing and usage.
- Ensure that the `.env` file is properly configured with valid API keys and paths before running the application.