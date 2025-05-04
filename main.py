import argparse
from time import sleep
import os
import os.path as osp
from typing import Literal, Optional, Union, List, Dict, Any
from typing_extensions import Annotated, TypedDict

from dotenv import load_dotenv

from langchain_chroma import Chroma

from googleapiclient.discovery import build
from googleapiclient.errors import HttpError


from config import Config, VALID_MIMETYPES, FILE_TYPE_MAP, VALID_YEARS, VALID_GEMINI_MODELS, VALID_OPENAI_MODELS


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


def get_embedding_function(args:Config):
    if args.emb_func == "google":
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        embedding_function = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004",
                                                          google_api_key=os.environ.get("GEMINI_API_KEY"))
    elif args.emb_func == "openai":
        from langchain_openai import OpenAIEmbeddings
        embedding_function = OpenAIEmbeddings(model="text-embedding-ada-002",
                                              openai_api_key=os.environ.get("OPENAI_API_KEY"))
    else:
        raise ValueError(f"Invalid embedding function: {args.emb_func}",
                         f"Valid options are: google, openai")
    return embedding_function


def get_llm(args:Config):
    if args.model in VALID_GEMINI_MODELS:
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(model=args.model, google_api_key=os.environ.get("GEMINI_API_KEY"))
    elif args.model in VALID_OPENAI_MODELS:
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=args.model, openai_api_key=os.environ.get("OPENAI_API_KEY"))
    else:
        raise ValueError(f"Invalid model: {args.model}",
                         f"Valid options are: {VALID_GEMINI_MODELS}, {VALID_OPENAI_MODELS}")


def get_gdrive_files(service, year:int, mime_type:str) -> list[dict]:
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


def get_file_bytes(service, file:dict) -> bytes:
    '''get the file content in bytes'''
    import io
    from googleapiclient.http import MediaIoBaseDownload
    try:
        # For Google Docs/Sheets/etc we need to export
        if file.get("mimeType").startswith("application/vnd.google-apps"):
            export_mime_type = {
                "application/vnd.google-apps.document": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                "application/vnd.google-apps.spreadsheet": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                "application/vnd.google-apps.presentation": "application/vnd.openxmlformats-officedocument.presentationml.presentation"
            }.get(file.get("mimeType"), "text/plain")

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


def get_parsed_elements(file:dict, file_bytes:bytes) -> str:
    '''read and partition the file content'''
    from unstructured.partition.auto import partition
    # filename = os.path.join(EXAMPLE_DOCS_DIRECTORY, "layout-parser-paper-fast.pdf")
    # with open(filename, "rb") as f:
    elements = partition(file=file_bytes, content_type=file.get("mimeType"))
    file_str = "\n\n".join([str(el) for el in elements]) # [:20])

    return file_str


def generate_questions_keywords_from(args:Config, file_str:str, file:dict, year:int) -> dict[dict]:
    from operator import itemgetter
    # from pydantic import BaseModel, Field
    class FinnishToEnglishTranslation(TypedDict):
        """Finnish-to-English translation of documents."""
        translation: Annotated[str, ..., """If the document is in Finnish, return its English translation. Otherwise, return 'None'. In any case, Do NOT return the original document."""]

    # from langchain_core.output_parsers import StrOutputParser
    # from langchain.chains.query_constructor.base import StructuredQueryOutputParser

    from langchain_core.prompts import ChatPromptTemplate

    translate_template = """You are an expert at translating Finnish documents into English.
        - If the document is in Finnish with a few English words and names, return its full English translation.
        - If the document is a mix of Finnish and English, return the full English translation.
        - If the document is already in English with a few Finnish words and names, then just return 'None'
        - Remember to keep the original names of humans and places in the document.
        - Make sure that the translation (if any) has the same structure as the original document.
        - Here is the document:\n\n{doc}"""
    translate_prompt = ChatPromptTemplate.from_template(translate_template)
    translate_llm = get_llm(args)
    # output_parser = StructuredQueryOutputParser.from_components()

    translate_chain = (
        translate_prompt
        | translate_llm.with_structured_output(schema=FinnishToEnglishTranslation)
        # | output_parser
    )

    class QuestionKeywordsPair(TypedDict):
        """A pair of a question and its related keywords."""
        question: Annotated[str, ..., """A question"""]
        keywords: Annotated[List[str], ..., """Related keywords."""]

    class FinalAnswer(TypedDict):
        """List of question and keywords pairs at general and specific levels."""
        general_question_keyword_pairs: Annotated[List[QuestionKeywordsPair], ..., """General questions from the whole document and their related keywords."""]
        specific_question_keyword_pairs: Annotated[List[QuestionKeywordsPair], ..., """Specific questions from the details of the document and their related keywords."""]
    
    question_keyword_template = """You are an expert at generating potential questions and keywords that one might ask from documents.
    Information about the documents:
    - The documents are from a database belonging to Aaltoes (Aalto Entrepreneurship Society) which is a student-led organization in Aalto University, Finland.
    - The documents are about the Aaltoes organization, its past activities and its past members.

    Information about the questions and keywords:
    - You generate both general questions (from the whole document) and specific questions (from the details of the document).
    - For each question, you also generate a list of important keywords that are relevant to the question.
    - These questions and keywords are to be stored in a database that is linked to the document datastore.
    - When an Aaltoes member or a visitor ask a question, the system will search for the most relevant question and keywords, and then returns the linked documents.
    - Therefore, the questions must be INDEPENDENT of each other, and must not refer to each other.
    - Questions also must not refer to a prior knowledge of the document (e.g. 'what was "the" trip about?', instead of "the" mention the trip name).
    - Questions must be about the Aaltoes organization, its past activities and its past members and formulated in the past tense.
    - Extract all the questions that are about these topics at both general and specific level.
    - If there are acronyms, words, or names of human and places that you are not familiar with, do not try to rephrase them.

    Here are some examples of the questions:
    - General questions:
        - What was the purpose of Aaltoes?
        - What trips did Aaltoes have in 2024?
        - What kind of partnership deals were signed with Aaltoes in 2021?
        - What happened in the X event?
        - Where was location of the X event?
        - Who was the president of Aaltoes in 2022?
        - How many members did Aaltoes have overall in 2022?
        - How many partners did Aaltoes have overall in 2021?
        - How much money did Aaltoes receive in general in 2021?
        - How much did Aaltoes spend in general in 2022?
        - What media coverage was Aaltoes featured in overall?

    - Specific questions:
        - What was the responsibilities of Maija in the X event?
        - What was the terms of the partnership deal with X company?
        - How much was the salary of the president of Aaltoes in 2019?
        - How much money did Aaltoes receive from just Aalto university in 2021?
        - How much did Aaltoes spend on just orientation week 2022?
        - When was the deadline for payment of the purchases of the X event?
        - Who from Aaltoes was/were responsible for the London trip?
        - Who was the main speaker of the X event?
    
    
    Given the following document, named "{name}" from year {year}, return as many question and keyword pairs as you can, based on the instructions:\n
    The document in original language:\n\n{doc}\n\n
    ------------------------\n\n
    The document translation to English (if the original is not in English):\n\n{translation}"""
    question_keyword_prompt = ChatPromptTemplate.from_template(question_keyword_template)

    question_keyword_llm = get_llm(args)

    overall_chain = (
        {"doc": itemgetter("doc"),
         "name": itemgetter("name"),
         "year": itemgetter("year"),
         "translation": itemgetter("doc") | translate_chain | itemgetter("translation")}
        | question_keyword_prompt
        | question_keyword_llm.with_structured_output(schema=FinalAnswer)
        # | StrOutputParser()
    )

    question_keys = overall_chain.invoke({"doc": file_str, "name": file.get("name"), "year": year})
    return question_keys


def index(args:Config):
    creds = gdrive_auth()
    # create drive api client
    service = build("drive", "v3", credentials=creds)

    embedding_function = get_embedding_function(args)

    from langchain_core.documents import Document
    db_persist_dir = os.environ.get(key="CHROMADB_PERSISTED_PATH", default="./chroma_db")
    vector_store = Chroma(collection_name="Questions_and_Keywords",
                          embedding_function=embedding_function,
                          persist_directory=db_persist_dir)

    years = VALID_YEARS if args.indx_years=="Full" else args.indx_years
    print("preparing docs ...")
    for year in years:
        for mime_type in VALID_MIMETYPES:
            print(f"    Searching for {mime_type} files from {year} ...")
            files_list = get_gdrive_files(service, year, mime_type)
            if len(files_list) == 0:
                print(f"        No {mime_type} files found for {year}")
                continue
            print(f"        Found {len(files_list)} {mime_type} files for {year}")
            
            for file in files_list:
                if file.get("name") == "test me":
                    continue
                file_bytes = get_file_bytes(service, file)
                file_str = get_parsed_elements(file, file_bytes)
                
                question_key_pairs = generate_questions_keywords_from(args, file_str, file, year)
                
                # Docs linked to summaries
                for question_level in ["general", "specific"]:
                    # for each question level, we will add the questions to the vector store
                    # and also add the metadata to the vector store
                    # but for generation, we will also give the metadata to the LLM 
                    questions_docs = [
                        Document(page_content=question_key_pair['question'],
                                 metadata={"name": file.get("name"),
                                            "id": file.get("id"),
                                            "year": year,
                                            "mimeType": mime_type,
                                            "level": question_level,
                                            # "keywords": question_key_pair['keywords'],
                                            # "potential_users": question_key_pair['potential_users'],
                                            })
                        for question_key_pair in question_key_pairs[f"{question_level}_question_keyword_pairs"]
                    ]
                    # ! ValueError: Expected metadata value for metadata in Chroma to be a str, int, float or bool, got keywords:List[str] which is a list in upsert.
                    # ! Try filtering complex metadata from the document using langchain_community.vectorstores.utils.filter_complex_metadata.
                    # TODO: for generation, we will also give the metadata to the LLM
                    # continue
                # continue
                    vector_store.add_documents(questions_docs)
            sleep(10)
    
    # return vector_store


def get_unique_docs(documents: list) -> list:
    """ Unique union of retrieved docs """
    # Get the files_list of all retrieved documents
    files_list = [doc.metadata for doc in documents]
    
    # Get unique documents, using dictionary comprehension
    unique_files_list = list({file['id']: file for file in files_list}.values())
    # Return
    return unique_files_list


def generate_retriever_reponse(args:Config, joined_files_str:str):
    from langchain_core.output_parsers import StrOutputParser
    # from langchain.chains.query_constructor.base import StructuredQueryOutputParser

    from langchain_core.prompts import ChatPromptTemplate

    retriever_template = """You are a helpful assistant that answers questions based on the documents provided.
    Information about the documents:
    - The documents are retrieved from a database belonging to Aaltoes (Aalto Entrepreneurship Society) which is a student-led organization in Aalto University, Finland.
    - The documents are about the Aaltoes organization, its past activities and its past members.
    - If you are unable to answer the question based on the retrieved documents, then you can ask the user to either:
        - rephrase the question,
        - increase the "top k" value.
        - or narrow down the search by filtering the documents by year. 
    
    - Although there might be some documents that are in Finnish language, you must answer the question only in English.
    - Write your answer in Markdown format, and cite the documents you used to answer the question by their number (Document i).

    - Here is the query:\n\n{query}\n
    - Here are the retrieved documents from the database:\n\n{docs}\n
    - Here is the query again:\n\n{query}"""
    
    retriever_prompt = ChatPromptTemplate.from_template(retriever_template)
    retriever_llm = get_llm(args)

    retriever_chain = (
        retriever_prompt
        | retriever_llm
        | StrOutputParser()
    )

    retriever_reponse = retriever_chain.invoke({"query": args.query, "docs": joined_files_str})
    return retriever_reponse


def retrieve(args:Config):    
    creds = gdrive_auth()
    # create drive api client
    service = build("drive", "v3", credentials=creds)

    embedding_function = get_embedding_function(args)

    from langchain_core.documents import Document
    db_persist_dir = os.environ.get(key="CHROMADB_PERSISTED_PATH", default="./chroma_db")
    vector_store = Chroma(collection_name="Questions_and_Keywords",
                          embedding_function=embedding_function,
                          persist_directory=db_persist_dir)
    
    if args.retr_year == "Full":
        docs = vector_store.similarity_search(query=args.query, k=args.top_k)
    else:
        docs = vector_store.similarity_search(query=args.query, k=args.top_k, 
                                              filter={"year": args.retr_year})

    files_list = get_unique_docs(docs)
    
    files_str = [f"""# Document {i} - "{file.get("name")}" from year {file.get("year")}:\n"""+get_parsed_elements(service, file) for i, file in enumerate(files_list, start=1)]

    joined_files_str = "\n\n----------\n\n".join(files_str)

    retriever_reponse = generate_retriever_reponse(args, joined_files_str)
    return retriever_reponse, files_list


def parse_args() -> Config:
    help_msg = """Aaltoes RAG with ChromaDB"""
    parser = argparse.ArgumentParser(description=help_msg, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--mode", type=str, default="retrieve", choices=["index", "retrieve"], help="index or retrieve")
    parser.add_argument("--model", type=str, default="gpt-4o", choices=VALID_GEMINI_MODELS + VALID_OPENAI_MODELS)
    parser.add_argument("--emb_func", type=str, default="openai", choices=["google", "openai"])  # feel free to add support for more embedding functions
    parser.add_argument("--indx_years", default=[2020])#"Full")

    parser.add_argument("--query", type=str, default="What was the purpose of Aaltoes?")
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--retr_year", default="Full", choices=[2024, 2023, 2022, 2021, 2020, "Full"])
    parser.add_argument("--file_type", default="Full", choices=["Full", "MSWords", "MSPP", "MSExcel", "PDF"])
    args = parser.parse_args()
    # Convert argparse.Namespace to Config
    return Config(
        mode=args.mode,
        model=args.model,
        emb_func=args.emb_func,
        indx_years=args.indx_years,
        query=args.query,
        top_k=args.top_k,
        retr_year=args.retr_year,
        file_type=args.file_type,
    )


def main():
    load_dotenv()
    args = parse_args()
    if args.mode == "retrieve":
        retriever_reponse, _ = retrieve(args)
        print(retriever_reponse)
    elif args.mode == "index":
        index(args)
    else:
        raise ValueError(f"Invalid mode: {args.mode}")


if __name__ == "__main__":
    main()