import gradio as gr
from config import APIConfig, VALID_GEMINI_MODELS, VALID_OPENAI_MODELS
from main import load_dotenv, retrieve_async


async def app(query, top_k, retr_year, llm):
    load_dotenv()
    args = APIConfig(
            mode="retrieve",
            model=llm,
            emb_func="openai",
            query=query,
            top_k=top_k,
            retr_year=retr_year
        )
    # args = APIConfig(query=query, top_k=top_k, retr_year=retr_year, model=llm, emb_func=emb_model)
    retriever_reponse, files_list = await retrieve_async(args)
    reference_files = [f"""- "{file.get('name')}", Year {file.get('year')}""" for file in files_list]
    reference_files = "\n".join(reference_files)
    return retriever_reponse, reference_files


examples = [
    ["What is Aaltoes’ main method of receiving funding?", None, None, None, None],
    ["What was the outcomes mentioned in the YYS Grant Application?", None, None, None, None],
    ["Which projects did YYS primarily supported in 2024?", None, None, None, None],
]


demo = gr.Interface(
    fn=app,
    inputs=[
        gr.Textbox(
            label="Search",
            lines=2,
            placeholder="Enter your search query ..."),
        gr.Slider(label="Top k queries", minimum=1, maximum=10, step=1, value=5),
        gr.Dropdown(label="Year", choices=["Full", "2024", "2022", "2021", "2020"]),
        gr.Dropdown(label="LLM", choices=VALID_GEMINI_MODELS + VALID_OPENAI_MODELS, type="value", value="gpt-4o")
    ],
    outputs=[
        gr.Markdown(label="RAG Response", container=True, show_label=True),
        gr.Markdown(label="References", container=True, show_label=True)
    ],
    title="Aaltoes AI Search",
    description="Enter a question about Aaltoes. AI will search for the answer among Aaltoes's Docs.",
    theme=gr.themes.Base(),
    concurrency_limit=5,
    examples=examples
)


if __name__ == "__main__":
    demo.launch(share=True)