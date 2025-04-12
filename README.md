# Aaltoes Agent

Aaltoes Agent is an AI-powered tool designed to interact with Google Drive using the `gdrive-mcp` server as a tool. It leverages the `smolagents` library.

## Features

- Interact with Google Drive using the `gdrive-mcp` server.
- Supports multiple LLM backends (`LiteLLM`, `HfApi`, `Transformers`).
- Easily configurable and extendable.

## Prerequisites

- Python 3.11 or higher.
- `uv` as the project manager.
- You also need my modified version of the `gdrive-mcp` server. You can find my fork [here](https://github.com/MMoshtaghi/servers). (Install it and build it)

## Quick Start

1. Clone the repository:
    ```bash
    git clone https://github.com/your-repo/aaltoes-agent.git
    cd aaltoes-agent
    ```

2. Set up your environment variables:
- Create a .env file in the project root.
- Add the following variables:
    - HF_API_KEY=<your_hf_api_key>
    - GROQ_API_KEY=<your_groq_api>
    - GDRIVE_MCP_SERVER_BUILD_PATH=<path_to_gdrive_mcp_server>

3. uv will install and run the project, just run the agent:
    ```bash
    uv run agent_gdrive.py
    ```