import argparse
import os
import os.path as osp
from typing import Literal, Optional, Tuple, Union, List, Dict, Any
from typing_extensions import Annotated, TypedDict

from dotenv import load_dotenv

from langchain_chroma import Chroma

from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

VALID_MIMETYPES = [
    "application/vnd.google-apps.document",
    "application/vnd.google-apps.presentation",
    # "application/vnd.google-apps.spreadsheet",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    # "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "application/pdf"
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
    SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]
    creds = None
    # The file token.json stores the user's access and refresh tokens, and is
    # created automatically when the authorization flow completes for the first
    # time.
    if osp.exists(".gdrive-server-credentials.json"):
        creds = Credentials.from_authorized_user_file(".gdrive-server-credentials.json", SCOPES)
    # If there are no (valid) credentials available, let the user log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                "gcp-oauth.keys.json", SCOPES
            )
            creds = flow.run_local_server(port=0)
            # Save the credentials for the next run
            with open(".gdrive-server-credentials.json", "w") as token:
                token.write(creds.to_json())
    return creds


def get_gdrive_files(service, year, mime_type) -> list:
    """Search file in drive location"""
    try:
        def get_the_year_folder_id(year):
            # search for the parent folder
            results = (
                service.files()
                .list(
                    q=f"name='{year}' and mimeType='application/vnd.google-apps.folder'",
                    fields="nextPageToken, files(id, name)",
                    supportsAllDrives=True,
                    includeItemsFromAllDrives=True,
                    corpora="drive", # in case you want to get access to a specific drive, change it to "drive"
                    driveId= "0ADYoYJTxCkiiUk9PVA", # add driveId to fields temporarily, to get the ID of a shared drive, then change corpora to 'drive'
                )
                .execute()
            )
            items = results.get("files", [])
            if not items:
                print(f"No folder found for {year}")
                return None
            else:
                return items[0].get("id")
        
        parent_folder = get_the_year_folder_id(year)
        files = []
        page_token = None
        while True:
            # pylint: disable=maybe-no-member
            response = (
                service.files()
                .list(
                    q= f"'{parent_folder}' in parents and mimeType='{mime_type}'",
                    fields="nextPageToken, files(id, name, mimeType, parents, driveId)",
                    supportsAllDrives=True,
                    includeItemsFromAllDrives=True,
                    corpora="drive", # in case you want to get access to a specific drive, change it to "drive"
                    driveId= "0ADYoYJTxCkiiUk9PVA", # add driveId to fields temporarily, to get the ID of a shared drive, then change corpora to 'drive'
                    pageToken=page_token,
                )
                .execute()
            )
            # for file in response.get("files", []):
            #     # Process change
                # print(f'        Found file: {file.get("name")}, {file.get("id")}')
            files.extend(response.get("files", []))
            page_token = response.get("nextPageToken", None)
            if page_token is None:
                break
        return files

    except HttpError as error:
        raise error
    except Exception as e:
        raise e


def get_file_bytes(service, file, mime_type):
    import io
    from googleapiclient.http import MediaIoBaseDownload
    try:
        # For Google Docs/Sheets/etc we need to export
        if mime_type.startswith("application/vnd.google-apps"):
            export_mime_type = {
                "application/vnd.google-apps.document": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                "application/vnd.google-apps.spreadsheet": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                "application/vnd.google-apps.presentation": "application/vnd.openxmlformats-officedocument.presentationml.presentation"
            }.get(mime_type, "text/plain")

            request = service.files().export_media(
                fileId=file.get("id"), mimeType=export_mime_type
            )
            downloaded_file = io.BytesIO()
            downloader = MediaIoBaseDownload(downloaded_file, request)
            done = False
            while done is False:
                status, done = downloader.next_chunk()
            
            return downloaded_file
        
        # For regular files, download content
        request = service.files().get_media(fileId=file.get("id"))
        downloaded_file = io.BytesIO()
        downloader = MediaIoBaseDownload(downloaded_file, request)
        done = False
        while done is False:
            status, done = downloader.next_chunk()

        return downloaded_file

    except HttpError as error:
        raise error
    except Exception as e:
        raise e


def get_parsed_elements(service, file, mime_type):
    '''get the file content in bytes
    read and partition the file content'''
    file_bytes = get_file_bytes(service, file, mime_type)

    from unstructured.partition.auto import partition
    # filename = os.path.join(EXAMPLE_DOCS_DIRECTORY, "layout-parser-paper-fast.pdf")
    # with open(filename, "rb") as f:
    elements = partition(file=file_bytes, content_type=mime_type)
    file_str = "\n\n".join([str(el) for el in elements]) # [:20])

    return file_str


def generate_questions_keywords_from(args, file_str):
    from operator import itemgetter
    # from pydantic import BaseModel, Field
    class FinnishToEnglishTranslation(TypedDict):
        """Finnish-to-English translation of documents."""

        translation: Annotated[str, ..., """If the document is in Finnish, return its English translation. Otherwise, return 'None'"""]

    # from langchain_core.output_parsers import StrOutputParser
    # from langchain.chains.query_constructor.base import StructuredQueryOutputParser

    from langchain_core.prompts import ChatPromptTemplate
    from langchain_google_genai import ChatGoogleGenerativeAI

    translate_template = """You are an expert at translating documents into English. \
        If the majority of the document is already in English, then just return 'None'.\
        Here is the document:\n\n{doc}"""
    translate_prompt = ChatPromptTemplate.from_template(translate_template)
    translate_llm = ChatGoogleGenerativeAI(model=args.model, google_api_key=os.environ.get("GEMINI_API_KEY"))
    # output_parser = StructuredQueryOutputParser.from_components()

    translate_chain = (
        translate_prompt
        | translate_llm.with_structured_output(schema=FinnishToEnglishTranslation)
        # | output_parser
    )

    class QuestionKeywordsPair(TypedDict):
        """Question and keywords pairs."""
        question_keywords_pair: Annotated[List[str], ..., """A pair of a question and its related keywords."""]

    class QuestionKeywordsList(TypedDict):
        """List of question and keywords pairs."""
        questions_keywords_pair_list: Annotated[List[QuestionKeywordsPair], ..., """A list of pairs of a question and its related keywords."""]

    class FinalAnswer(TypedDict):
        """List of question and keywords pairs at general and specific levels."""
        general_question_keyword_pairs: Annotated[QuestionKeywordsList, ..., """General questions from the whole document and their related keywords."""]
        specific_question_keyword_pairs: Annotated[QuestionKeywordsList, ..., """Specific questions from the details of the document and their related keywords."""]
    
    question_keyword_template = """You are an expert at generating potential questions that one might ask from documents. \
    You generate both general questions (from the whole document) and specific questions (from the details of the document). \
    For each question, you also generate a list of keywords that are relevant to the question. \
    The keywords are to be used to search a database of documents. \
    If there are acronyms, words, or names of human and places that you are not familiar with, do not try to rephrase them.
    
    Given the following document, return a list of question and keyword pairs as instructed:\n\n
    Document in original language:\n\n{doc}\n\n
    ------------------------\n\n
    Document translation to English (if the original is not in English):\n\n{translation}"""
    question_keyword_prompt = ChatPromptTemplate.from_template(question_keyword_template)

    question_keyword_llm = ChatGoogleGenerativeAI(model=args.model, google_api_key=os.environ.get("GEMINI_API_KEY"))

    overall_chain = (
        {"doc": itemgetter("doc"),
         "translation": itemgetter("doc") | translate_chain | itemgetter("translation")}
        | question_keyword_prompt
        | question_keyword_llm.with_structured_output(schema=FinalAnswer)
        # | StrOutputParser()
    )

    question_keys = overall_chain.invoke({"doc": file_str})
    return question_keys


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
    # create drive api client
    service = build("drive", "v3", credentials=creds)

    from langchain_google_genai import GoogleGenerativeAIEmbeddings
    embedding_function = GoogleGenerativeAIEmbeddings(model=args.emb_func, google_api_key=os.environ.get("GEMINI_API_KEY"))

    from langchain_core.documents import Document
    db_persist_dir = os.environ.get(key="CHROMADB_PERSISTED_PATH", default="./chroma_db")
    vector_store = Chroma(collection_name="summaries",
                          embedding_function=embedding_function,
                          persist_directory=db_persist_dir)

    print("preparing docs ...")
    for year in VALID_YEARS:
        for mime_type in VALID_MIMETYPES:
            print(f"    Searching for {mime_type} files from {year} ...")
            files_list = get_gdrive_files(service, year, mime_type)
            if len(files_list) == 0:
                print(f"        No {mime_type} files found for {year}")
                continue
            print(f"        Found {len(files_list)} {mime_type} files for {year}")
            for file in files_list:
                '''
                get the list file ids
                one by one:
                    get the file content in base64
                    read and partition the file content
                    generate questions
                    delete the file
                '''
                file_str = get_parsed_elements(service, file, mime_type)
                
                question_key_pairs = generate_questions_keywords_from(args, file_str)
                continue
                
                # Docs linked to summaries
                questions_docs = [
                    Document(page_content=q, metadata={"file_name": file.get("name"),
                                                       "file_id": file.get("id"),
                                                       "year": year,
                                                       "mime_type": mime_type,
                                                       "keywords": keyword,
                                                       })
                    for q, keyword in question_key_pairs
                ]
                vector_store.add_documents(questions_docs)
    
    return vector_store

if __name__ == "__main__":
    vector_store = index()

    query = "Memory in agents"
    sub_docs = vector_store.similarity_search(query,k=1)
    ''' sub_docs[0]:
    Document(page_content='The document discusses the concept of building autonomous agents powered by Large Language Models (LLMs) as their core controllers. It covers components such as planning, memory, and tool use, along with case studies and proof-of-concept examples like AutoGPT and GPT-Engineer. Challenges like finite context length, planning difficulties, and reliability of natural language interfaces are also highlighted. The document provides references to related research papers and offers a comprehensive overview of LLM-powered autonomous agents.', metadata={'doc_id': 'cf31524b-fe6a-4b28-a980-f5687c9460ea'}) '''

