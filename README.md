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

## Quick Start

1. **Index Documents**:
    To index documents from Google Drive, run:
    ```bash
    uv run main.py --mode index
    ```

2. **Search and Retrieve**:
    Use the Gradio interface to input your query, select parameters like year, model, and embedding function, and retrieve answers along with document references. To launch the Gradio interface, run:
    ```bash
    uv run gr_app.py
    ```

    you can also use CLI, run:
    ```bash
    uv run main.py --mode retrieve --query "your query"
    ```