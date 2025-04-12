import argparse
import os
from dotenv import load_dotenv

# from langchain_openai import OpenAIEmbeddings
from smolagents import GradioUI, LiteLLMModel, Tool, ToolCollection
from smolagents.agents import CodeAgent #, ToolCallingAgent

from mcp import StdioServerParameters


def parse_args():
    help_msg = """\
Agentic RAG with ChromaDB
"""
    parser = argparse.ArgumentParser(description=help_msg, formatter_class=argparse.RawTextHelpFormatter)
    # Essential Args
    # For anthropic: change args below to 'LiteLLM', 'anthropic/claude-3-5-sonnet-20240620' and "ANTHROPIC_API_KEY"
    parser.add_argument("--model_src", type=str, default="LiteLLM", choices=["HfApi", "LiteLLM", "Transformers"])
    parser.add_argument("--model", type=str, default="groq/qwen-2.5-coder-32b")
    parser.add_argument("--LiteLLMModel_API_key_name", type=str, default="GROQ_API_KEY")
    parser.add_argument(
        "--emb_func",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
        choices=["sentence-transformers/all-MiniLM-L6-v2"],
    )  # feel free to add support for more embedding functions
    parser.add_argument("--persist_dir", type=str, default="./chroma_db", help="Path to the persisted vector DB")

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    # Choose which LLM engine to use!
    if args.model_src == "HfApi":
        from smolagents import HfApiModel

        # You can choose to not pass any model_id to HfApiModel to use a default free model
        model = HfApiModel(model_id=args.model, token=os.environ.get("HF_API_KEY"))

    elif args.model_src == "Transformers":
        from smolagents import TransformersModel

        model = TransformersModel(model_id=args.model)

    elif args.model_src == "LiteLLM":
        model = LiteLLMModel(model_id=args.model, api_key=os.environ.get(args.LiteLLMModel_API_key_name))
    else:
        raise ValueError('Choose the models source from ["HfApi", "LiteLLM", "Transformers"]')

    server_parameters = StdioServerParameters(
        command= os.environ.get("GDRIVE_MCP_SERVER_BUILD_PATH"),
        args= [
        "-y"
      ]
    )

    with ToolCollection.from_mcp(server_parameters=server_parameters) as tool_collection:
        agent = CodeAgent(tools=[*tool_collection.tools], add_base_tools=False,
                          model=model, max_steps=3, verbosity_level=2)

        # GradioUI(agent).launch()

        agent_output = agent.run("Please find a file about 'Pitch Aaltoes' in my Google Drive, and then summarize it.")
        print("Final output:")
        print(agent_output)


if __name__ == "__main__":
    load_dotenv()
    main()
