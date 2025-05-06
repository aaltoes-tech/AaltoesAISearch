import gradio as gr
from config import Config, VALID_GEMINI_MODELS, VALID_OPENAI_MODELS
from main import load_dotenv, retrieve_async
import argparse

# @dataclass
# class Config:
#     query : str
#     top_k : int = 5
#     retr_year : int | str = "Full"
#     model : str = "gpt-4o"
#     emb_func : str = "openai"
#     mode : str = "retrieve"


async def app(query, top_k, retr_year, llm, emb_model):
    load_dotenv()
    args = Config(
            mode="retrieve",
            model=llm,
            emb_func=emb_model,
            query=query,
            top_k=top_k,
            retr_year=retr_year
        )
    # args = Config(query=query, top_k=top_k, retr_year=retr_year, model=llm, emb_func=emb_model)
    retriever_reponse, files_list = await retrieve_async(args)
    reference_files = [f"""- Document {i} : "{file.get("name")}" from year {file.get("year")}"""for i, file in enumerate(files_list, start=1)]
    reference_files = "\n".join(reference_files)
    return retriever_reponse, reference_files


demo = gr.Interface(
    fn=app,
    inputs=[
        gr.Textbox(
            label="Search",
            lines=2,
            placeholder="Enter your search query ..."),
        gr.Slider(label="Top k queries", minimum=1, maximum=10, step=1, value=5),
        gr.Dropdown(label="Year", choices=["Full", "2024", "2022", "2021", "2020"]),
        gr.Dropdown(label="LLM", choices=VALID_GEMINI_MODELS + VALID_OPENAI_MODELS, type="value", value="gpt-4o"),
        gr.Dropdown(label="Embedding Model", choices=["google", "openai"], type="value", value="openai"),
    ],
    outputs=[
        gr.Markdown(label="RAG Response", container=True, show_label=True),
        gr.Markdown(label="References", container=True, show_label=True)
    ],
    title="Aaltoes AI Search",
    description="Enter a question about Aaltoes. AI will search for the answer among Aaltoes's Docs.",
    theme=gr.themes.Base(),
    concurrency_limit=5
)


if __name__ == "__main__":
    demo.launch()