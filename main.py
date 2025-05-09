import argparse
import time
import os
import os.path as osp
from typing import Literal, Optional, Union, Any
from typing_extensions import Annotated, TypedDict

import asyncio
from concurrent.futures import ProcessPoolExecutor
import pandas as pd

from dotenv import load_dotenv

from langchain_chroma import Chroma

from googleapiclient.discovery import build
from googleapiclient.errors import HttpError


from config import APIConfig, PrfConfig, VALID_MIMETYPES, FILE_TYPE_MAP, VALID_YEARS, VALID_GEMINI_MODELS, VALID_OPENAI_MODELS

import logging
logging.basicConfig(level=logging.ERROR, format="%(levelname)s - %(message)s")

prf_config = PrfConfig(max_num_cores=8, max_concurrent_indexing=16)


class CheckpointTracker:
    """A class to track indexed files and enable checkpoint recovery"""
    def __init__(self, checkpoint_file:str="indexing_checkpoint.csv"):
        self.checkpoint_file = checkpoint_file
        if osp.exists(self.checkpoint_file):
            # Load existing completed files from the checkpoint file
            self.completed_files = pd.read_csv(self.checkpoint_file)
        else:
            # Create a new DataFrame if the checkpoint file does not exist
            self.completed_files = pd.DataFrame(columns=["file_id", "file_name", "status", "timestamp"])

        self.lock = asyncio.Lock()

    async def mark_completed(self, file:dict, status:str):
        try:
            async with self.lock:
                new_row = pd.DataFrame({
                    "file_id": [file.get("id")],
                    "file_name": [file.get("name")],
                    "status": [status],
                    "timestamp": [pd.Timestamp.now()]
                })
                # Append the new row to the existing DataFrame, avoiding duplicates
                self.completed_files = pd.concat([
                    self.completed_files[self.completed_files["file_id"] != file.get("id")],
                    new_row
                ])
        
                if status == "failed":
                    # Save checkpoint on error, not on success to speed up indexing
                    self.completed_files.to_csv(self.checkpoint_file, index=False)
        except Exception as e:
            logging.error(f"Error in mark_completed for file {file.get('name')}: {e}")
            raise e
    
    def save_checkpoint_to_csv_no_lock(self):
        """Save the completed files to a CSV file"""
        try:
            self.completed_files.to_csv(self.checkpoint_file, index=False)
        except Exception as e:
            logging.error(f"Error saving checkpoint to CSV: {e}")
            raise e

    def get_unprocessed_files(self, files_list:list[dict]) -> list[dict]:
        """Get the list of unprocessed files"""
        # Filter out completed files
        completed_files = self.completed_files[self.completed_files["status"] == "completed"]["file_id"].values
        return [ file for file in files_list if file.get("id") not in completed_files]


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


def get_embedding_function(args:APIConfig):
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


def get_llm(args:APIConfig):
    if args.model in VALID_GEMINI_MODELS:
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(model=args.model, google_api_key=os.environ.get("GEMINI_API_KEY"))
    elif args.model in VALID_OPENAI_MODELS:
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=args.model, openai_api_key=os.environ.get("OPENAI_API_KEY"))
    else:
        raise ValueError(f"Invalid model: {args.model}",
                         f"Valid options are: {VALID_GEMINI_MODELS}, {VALID_OPENAI_MODELS}")


def get_gdrive_files(args: APIConfig, creds) -> tuple[list[dict], dict]:
    """Search file in drive location"""
    try:
        # create drive api client
        service = build("drive", "v3", credentials=creds)

        def get_year_folders(years:list[str]) -> list[dict]:
            # search for the parent folder
            names_filter = " or ".join([f"name='{year}'" for year in years])
            folders = []
            page_token = None
            while True:
                # pylint: disable=maybe-no-member
                response = (
                    service.files()
                    .list(
                        q=f"({names_filter}) and mimeType='application/vnd.google-apps.folder'",
                        fields="nextPageToken, files(id, name)",
                        supportsAllDrives=True,
                        includeItemsFromAllDrives=True,
                        corpora="drive", # in case you want to get access to a specific drive, change it to "drive"
                        driveId= "0ADYoYJTxCkiiUk9PVA", # add driveId to fields temporarily, to get the ID of a shared drive, then change corpora to 'drive'
                        pageToken=page_token,
                    )
                    .execute()
                )
                folders.extend(response.get("files", []))
                page_token = response.get("nextPageToken", None)
                if page_token is None:
                    break
            
            if len(folders)==0:
                print(f"No folder found for {years=}")
                return None
            else:
                return folders
        
        years = VALID_YEARS if args.indx_years=="Full" else args.indx_years
        parent_folders = get_year_folders(years)
        if parent_folders is None:
            print(f"No folders found for {years=}")
            return [], {}
        
        parent_folders_filter = [f"""'{parent_folder.get("id")}' in parents""" for parent_folder in parent_folders]
        parent_folders_filter = " or ".join(parent_folders_filter)

        if args.file_type == "Full":
            mime_types_filter = [f"""mimeType='{mime_type}'""" for mime_type in VALID_MIMETYPES]
            mime_types_filter = " or ".join(mime_types_filter)
        else:
            mime_types_filter = f"""mimeType='{FILE_TYPE_MAP.get(args.file_type)}'"""
        
        query = f"""({parent_folders_filter}) and ({mime_types_filter})"""
        files = []
        page_token = None
        while True:
            # pylint: disable=maybe-no-member
            response = (
                service.files()
                .list(
                    q= query,
                    fields="nextPageToken, files(id, name, mimeType, parents, driveId)",
                    supportsAllDrives=True,
                    includeItemsFromAllDrives=True,
                    corpora="drive", # in case you want to get access to a specific drive, change it to "drive"
                    driveId= "0ADYoYJTxCkiiUk9PVA", # add driveId to fields temporarily, to get the ID of a shared drive, then change corpora to 'drive'
                    pageToken=page_token,
                )
                .execute()
            )
            files.extend(response.get("files", []))
            page_token = response.get("nextPageToken", None)
            if page_token is None:
                break
        
        
        id_to_year_map = {folder.get("id"): folder.get("name") for folder in parent_folders}
        return files, id_to_year_map

    except HttpError as e:
        logging.error(f"HttpError getting gdrive files list: {e}")
        return [], {}
    except Exception as e:
        logging.error(f"Error getting gdrive files list: {e}")
        return [], {}


async def get_file_bytes_async(creds, file: dict) -> bytes:
    '''get the file content in bytes asynchronously'''
    import io
    from googleapiclient.http import MediaIoBaseDownload
    # create drive api client
    service = build("drive", "v3", credentials=creds)
    # create event loop
    loop = asyncio.get_event_loop()
    
    try:
        # A function to handle synchronous Google API calls
        def _download_file():
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
            else:
                # For regular files, download content
                request = service.files().get_media(fileId=file.get("id"))
                
            downloaded_file = io.BytesIO()
            downloader = MediaIoBaseDownload(downloaded_file, request)
            done = False
            while done is False:
                status, done = downloader.next_chunk()
            downloaded_file.seek(0)  # Reset the file pointer
            return downloaded_file
            
        # Run the synchronous Google API call in a thread pool
        return await loop.run_in_executor(None, _download_file)
    except HttpError as e:
        logging.error(f"HttpError downloading file {file.get('name')}: {e}")
        raise e
    except Exception as e:
        logging.error(f"Error downloading file {file.get('name')}: {e}")
        raise e


def get_parsed_elements(file:dict, file_bytes:bytes) -> str:
    '''read and partition the file content'''
    from unstructured.partition.auto import partition
    try:
        elements = partition(file=file_bytes, content_type=file.get("mimeType"))
        file_str = "\n\n".join([str(el) for el in elements]) # [:20])
        return file_str
    except Exception as e:
        logging.error(f"Error in get_parsed_elements for file {file.get('name')}: {e}")
        return None # Don't raise an error here, otherwise it kills the whole common Process Pool


async def generate_questions_keywords_async(args:APIConfig, file_str:str, file:dict) -> dict[dict]:
    from operator import itemgetter
    # from pydantic import BaseModel, Field
    class FinnishToEnglishTranslation(TypedDict):
        """Finnish-to-English translation of documents."""
        translation: Annotated[str, ..., """If the document is in Finnish, return its English translation. Otherwise, return 'None'. In any case, Do NOT return the original document."""]

    # from langchain_core.output_parsers import StrOutputParser
    # from langchain.chains.query_constructor.base import StructuredQueryOutputParser

    from langchain_core.prompts import ChatPromptTemplate

    try:
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
            keywords: Annotated[list[str], ..., """Related keywords."""]

        class FinalAnswer(TypedDict):
            """List of question and keywords pairs at general and specific levels."""
            general_question_keyword_pairs: Annotated[list[QuestionKeywordsPair], ..., """General questions from the whole document and their related keywords."""]
            specific_question_keyword_pairs: Annotated[list[QuestionKeywordsPair], ..., """Specific questions from the details of the document and their related keywords."""]
        
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
        - Some documents may be vague and clear what they are about, so you can use the name and year of the document in the questions to clarify the question.
        - If there are acronyms, words, or names of human and places that you are not familiar with, do not try to rephrase them.

        Here are some examples of the questions that we are interested in:
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

        question_keys = await overall_chain.ainvoke({"doc": file_str, "name": file.get("name"), "year": file.get("year_parent_name")})
        return question_keys
    
    except Exception as e:
        logging.error(f"Error in generate_questions_keywords_async for file {file.get('name')}: {e}")
        raise e


async def add_to_vector_store(file, question_key_pairs, vector_store):
    """Add questions to vector store"""
    from langchain_core.documents import Document
    try:
        documents = []
        
        # Process both general and specific questions
        for question_level in ["general", "specific"]:
            questions_docs = [
                Document(
                    page_content=question_key_pair['question'],
                    metadata={
                        "name": file.get("name"),
                        "id": file.get("id"),
                        "year": file.get("year_parent_name"),
                        "mimeType": file.get("mimeType"),
                        "level": question_level,
                    }
                )
                for question_key_pair in question_key_pairs[f"{question_level}_question_keyword_pairs"]
            ]
            documents.extend(questions_docs)
        
        # Add documents to vector store in a single batch
        # Run in executor since vector store operations might be blocking
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            lambda: vector_store.add_documents(documents)
        )

    except Exception as e:
        logging.error(f"Error adding to vector store for file {file.get('name')}: {e}")
        raise e


async def index_process_single_file(creds, file:dict, args:APIConfig, id_to_year_map:dict,
                                    process_semaphore: asyncio.Semaphore, process_pool:ProcessPoolExecutor, 
                                    vector_store:Any, checkpoint_tracker:CheckpointTracker):
    """Process a single file through the entire pipeline with semaphore control"""
    try:
        # Acquire the process semaphore to limit overall concurrent processing
        async with process_semaphore:
            # Add parent name (year) to file
            file["year_parent_name"] = id_to_year_map.get(file.get("parents")[0], None)
            
            # Download file with concurrency control
            # async with download_semaphore:
            # print(f"Downloading {file.get('name')}...")
            file_bytes = await get_file_bytes_async(creds, file)
            
            # Parse file content (CPU-bound) in process pool
            # print(f"Parsing {file.get('name')}...")
            file_str = await asyncio.get_event_loop().run_in_executor(
                process_pool,
                get_parsed_elements,
                file,
                file_bytes
            )
            if file_str is None:
                raise Exception()
            
            # Generate questions and keywords (I/O-bound LLM call)
            # print(f"Generating questions for {file.get('name')}...")
            question_key_pairs = await generate_questions_keywords_async(args, file_str, file)
            
            # Add to vector store
            # print(f"Adding to vector store: {file.get('name')}...")
            await add_to_vector_store(file, question_key_pairs, vector_store)
            
            # Mark file as completed in the checkpoint tracker
            await checkpoint_tracker.mark_completed(file, "completed")

            print(f"Completed indexing {file.get('name')}")
            return {"file_id": file.get("id"), "file_name": file.get("name"), "status": "completed"}
        
    except Exception as e:
        logging.error(f"Error in process_single_file for {file.get('name')}: {e}")
        await checkpoint_tracker.mark_completed(file, "failed")
        return {"file_id": file.get("id"), "file_name": file.get("name"), "status": "failed"}


async def index_async(args:APIConfig):
    """Asynchronous version of the index function"""
    print("Starting async indexing...")
    
    # Initialize services
    creds = gdrive_auth()

    embedding_function = get_embedding_function(args)

    # Initialize vector store
    db_persist_dir = os.environ.get(key="CHROMADB_PERSISTED_PATH", default="./chroma_db")
    vector_store = Chroma(collection_name="Questions_and_Keywords",
                          embedding_function=embedding_function,
                          persist_directory=db_persist_dir)

    # Get files info
    print("Preparing docs ...")
    files_list, id_to_year_map = get_gdrive_files(args, creds)
    if len(files_list) == 0:
        print("No files found to index. Exiting.")
        return None
    print(f"Found {len(files_list)} files")

    checkpoint_tracker = CheckpointTracker()  # Initialize checkpoint tracker

    # Filter out already processed files if not explicitly requested to process all
    if args.resume_checkpoint:
        unprocessed_files = checkpoint_tracker.get_unprocessed_files(files_list)
        print(f"Resuming from checkpoint: {len(files_list) - len(unprocessed_files)} files already indexed, {len(unprocessed_files)} remaining.")
        files_to_index = unprocessed_files
    else:
        print(f"Indexing all {len(files_list)} files regardless of checkpoint.")
        files_to_index = files_list
    
    # Create semaphores for concurrency control
    process_semaphore = asyncio.Semaphore(prf_config.max_concurrent_indexing)
    # Create process pool for CPU-bound parsing operations
    process_pool = ProcessPoolExecutor(max_workers= prf_config.max_num_cores)

    try:
        # Create a task for each file
        tasks = []
        for file in files_to_index:
            if file.get("name") == "test me":
                continue
                
            task = asyncio.create_task(
                index_process_single_file(
                    creds, 
                    file, 
                    args,
                    id_to_year_map,
                    process_semaphore, 
                    process_pool, 
                    vector_store,
                    checkpoint_tracker
                )
            )
            tasks.append(task)
        
        # Process all files concurrently with controlled concurrency
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Save checkpoint after all tasks are completed
        checkpoint_tracker.save_checkpoint_to_csv_no_lock()

        # Check for any exceptions
        success_count = 0
        for result in results:
            if isinstance(result, Exception):
                logging.error(f"Task failed with exception: {result}")
            elif result.get("status") == "failed":
                logging.warning(f"Failed to index file {result.get('file_name')}")
            else:
                success_count += 1
        
        print(f"Indexing completed. Successfully processed {success_count} out of {len(tasks)} files.")
        return vector_store
    
    except Exception as e:
        logging.error(f"Error in index_async: {e}")
        raise e
        
    finally:
        # Clean up resources
        process_pool.shutdown()


def index(args: APIConfig):
    """Wrapper for the async index function"""
    return asyncio.run(index_async(args), debug=False)


def get_unique_docs(documents: list) -> list:
    """ Unique union of retrieved docs """
    # Get the files_list of all retrieved documents
    files_list = [doc.metadata for doc in documents]
    
    # Get unique documents, using dictionary comprehension
    unique_files_list = list({file['id']: file for file in files_list}.values())
    # Return
    return unique_files_list


async def retrieve_process_single_file(creds, file, process_semaphore, process_pool):
    try:
        # Acquire the process semaphore to limit overall concurrent processing
        async with process_semaphore:
            # Download file content
            print(f"Downloading {file.get('name')}...")
            file_bytes = await get_file_bytes_async(creds, file)

            # Parse file content (CPU-bound)
            print(f"Parsing {file.get('name')}...")
            file_str = await asyncio.get_event_loop().run_in_executor(
                process_pool, get_parsed_elements, file, file_bytes
            )
            if file_str is None:
                raise Exception()

            # Format the parsed content for aggregation
            return f"""# Document reference "{file.get('name')}", Year {file.get('year')}:\n{file_str}"""
    except Exception as e:
        logging.error(f"Error processing file {file.get('name')}: {e}")
        return None


async def generate_retriever_reponse_async(args:APIConfig, joined_files_str:str):
    from langchain_core.output_parsers import StrOutputParser
    # from langchain.chains.query_constructor.base import StructuredQueryOutputParser

    from langchain_core.prompts import ChatPromptTemplate

    try:
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

        retriever_reponse = await retriever_chain.ainvoke({"query": args.query, "docs": joined_files_str})
        return retriever_reponse
    except Exception as e:
        logging.error(f"Error in generate_retriever_reponse_async: {e}")
        raise e


async def retrieve_async(args:APIConfig):
    """Retrieve documents from the vector store"""
    creds = gdrive_auth()

    embedding_function = get_embedding_function(args)
    # Initialize vector store
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

    # Create semaphores for concurrency control
    process_semaphore = asyncio.Semaphore(prf_config.max_concurrent_indexing)
    # Create process pool for CPU-bound operations
    process_pool = ProcessPoolExecutor(max_workers= prf_config.max_num_cores)

    try:
        # Create tasks for processing files
        tasks = [retrieve_process_single_file(
            creds, file, process_semaphore, process_pool
            ) for file in files_list ]
        parsed_files = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter out None results and aggregate parsed content
        parsed_files = [content for content in parsed_files if content is not None]
        joined_files_str = "\n\n----------\n\n".join(parsed_files)

        # Generate the final response using the LLM
        print("Generating final response...")
        retriever_response = await generate_retriever_reponse_async(args, joined_files_str)

        return retriever_response, files_list
    except Exception as e:
        logging.error(f"Error in retrieve_async: {e}")
        raise e
    finally:
        # Clean up resources
        process_pool.shutdown()


def retrieve(args:APIConfig):
    """Wrapper for the async retrieve function"""
    return asyncio.run(retrieve_async(args), debug=False)


def parse_args() -> APIConfig:
    help_msg = """Aaltoes RAG with ChromaDB"""
    parser = argparse.ArgumentParser(description=help_msg, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--mode", type=str, default="index", choices=["index", "retrieve"], help="index or retrieve")
    parser.add_argument("--model", type=str, default="gpt-4o", choices=VALID_GEMINI_MODELS + VALID_OPENAI_MODELS)
    parser.add_argument("--emb_func", type=str, default="openai", choices=["google", "openai"])  # feel free to add support for more embedding functions
    parser.add_argument("--indx_years", default=["2024"]) #"Full")
    parser.add_argument("--file_type", default="PDF", choices=["Full", "MSWords", "MSPP", "MSExcel", "PDF"])
    parser.add_argument("--resume_checkpoint", type=bool, default=True, help="Resume from checkpoint")

    parser.add_argument("--query", type=str, default="What was the purpose of Aaltoes?")
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--retr_year", default="Full", choices=["2024", "2023", "2022", "2021", "2020", "Full"])
    args = parser.parse_args()
    # Convert argparse.Namespace to APIConfig
    return APIConfig(
        mode=args.mode,
        model=args.model,
        emb_func=args.emb_func,
        indx_years=args.indx_years,
        query=args.query,
        top_k=args.top_k,
        retr_year=args.retr_year,
        file_type=args.file_type,
        resume_checkpoint=args.resume_checkpoint
    )


def main():
    load_dotenv()
    args = parse_args()
    if args.mode == "retrieve":
        retriever_reponse, references = retrieve(args)
        print(retriever_reponse)
        print("References: {references}")
    elif args.mode == "index":
        index(args)
    else:
        raise ValueError(f"Invalid mode: {args.mode}")


if __name__ == "__main__":
    main()