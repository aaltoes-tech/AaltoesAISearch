import argparse
import os
import os.path as osp

from dotenv import load_dotenv
from langchain_chroma import Chroma


VALID_MIMETYPES = [
    "application/vnd.google-apps.document",
    "application/vnd.google-apps.drawing",
    "application/vnd.google-apps.presentation",
    "application/vnd.google-apps.script",
    "application/vnd.google-apps.spreadsheet",
    "application/vnd.google-apps.form",
    "application/vnd.google-apps.folder",
    "application/vnd.google-apps.map",
    "application/vnd.google-apps.photo",
    "application/vnd.google-apps.site",
]

VALID_YEARS = [
    "2020",
    "2021",
    "2022",
    "2023",
    "2024",
]  # Add more years as needed


def gdrive_auth():
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow

    # If modifying these scopes, delete the file token.json.
    SCOPES = ["https://www.googleapis.com/auth/drive.metadata.readonly"]
    creds = None
    # The file token.json stores the user's access and refresh tokens, and is
    # created automatically when the authorization flow completes for the first
    # time.
    if osp.exists("token.json"):
        creds = Credentials.from_authorized_user_file("token.json", SCOPES)
    # If there are no (valid) credentials available, let the user log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                "credentials.json", SCOPES
            )
            creds = flow.run_local_server(port=0)
            # Save the credentials for the next run
            with open("token.json", "w") as token:
                token.write(creds.to_json())
    return creds


def get_gdrive_files(creds, year) -> list:
    """Search file in drive location

    Load pre-authorized user credentials from the environment.
    TODO(developer) - See https://developers.google.com/identity
    for guides on implementing OAuth2 for the application.
    """
    from googleapiclient.discovery import build
    from googleapiclient.errors import HttpError

    try:
        # create drive api client
        service = build("drive", "v3", credentials=creds)
        
        def get_the_parent_folder(year):
            # search for the parent folder
            results = (
                service.files()
                .list(
                    q=f"name='{year}' and mimeType='application/vnd.google-apps.folder'",
                    fields="nextPageToken, files(id, name)",
                    supportsAllDrives=True,
                    includeItemsFromAllDrives=True,
                    corpora="user",
                    spaces="drive",
                )
                .execute()
            )
            items = results.get("files", [])
            if not items:
                print(f"No folder found for {year}")
                return None
            else:
                return items[0].get("id")
        
        files = []
        page_token = None
        while True:
            # pylint: disable=maybe-no-member
            response = (
                service.files()
                .list(
                    q= f"'{parent_folder}' in parents",
                    fields="nextPageToken, files(id, name, mimeType, parents)",
                    supportsAllDrives=True,
                    includeItemsFromAllDrives=True,
                    corpora="user", # in case you want to get access to a specific drive, change it to "drive"
                    # driveId: "YOUR_DRIVE_ID" # add driveId to fields temporarily, to get the ID of a shared drive, then change corpora to 'drive'
                    spaces="drive",
                    
                    pageToken=page_token,
                )
                .execute()
            )
            for file in response.get("files", []):
                # Process change
                print(f'Found file: {file.get("name")}, {file.get("id")}')
            files.extend(response.get("files", []))
            page_token = response.get("nextPageToken", None)
            if page_token is None:
                break

    except HttpError as error:
        print(f"An error occurred: {error}")
        files = None

    return files


def read_gdrive_files(files_list):
    pass
    # docs = loader.load()
    # docs.extend(loader.load())

    # return docs


def summerize_docs(args, docs):
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_google_genai import ChatGoogleGenerativeAI


    llm = ChatGoogleGenerativeAI(model=args.model, google_api_key=os.environ.get("GEMINI_API_KEY"))

    chain = (
        {"doc": lambda x: x.page_content}
        | ChatPromptTemplate.from_template("Summarize the following document:\n\n{doc}")
        | llm
        | StrOutputParser()
    )

    summaries = chain.batch(docs, {"max_concurrency": 5})
    return summaries


def parse_args():
    help_msg = """Agentic RAG with ChromaDB"""
    parser = argparse.ArgumentParser(description=help_msg, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--model", type=str, default="gemini-2.0-flash")
    parser.add_argument(
        "--emb_func",
        type=str,
        default="gemini-embedding-exp-03-07",
        choices=["gemini-embedding-exp-03-07"],
    )  # feel free to add support for more embedding functions

    args = parser.parse_args()
    return args


def index():
    load_dotenv()
    args = parse_args()
    
    creds = gdrive_auth()

    from langchain_google_genai import GoogleGenerativeAIEmbeddings
    embedding_function = GoogleGenerativeAIEmbeddings(model=args.emb_func, google_api_key=os.environ.get("GEMINI_API_KEY"))

    from langchain_core.documents import Document
    db_persist_dir = os.environ.get(key="CHROMADB_PERSISTED_PATH", default="./chroma_db")
    vector_store = Chroma(collection_name="summaries",
                          embedding_function=embedding_function,
                          persist_directory=db_persist_dir)

    print("preparing docs ...\n")
    for year in VALID_YEARS:
        for mimetype in VALID_MIMETYPES:
            print(f"Searching for {mimetype} files from {year} ...")
            files_list = get_gdrive_files(creds)
            docs = read_gdrive_files(files_list)
            summaries = summerize_docs(args, docs)

            # Docs linked to summaries
            summary_docs = [
                Document(page_content=s, metadata={"year": year, "mimetype": mimetype})
                for i, s in enumerate(summaries)
            ]
            vector_store.add_documents(summary_docs)

if __name__ == "__main__":
    index()
