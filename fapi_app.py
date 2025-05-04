import argparse
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Union
import os
from config import Config, VALID_GEMINI_MODELS, VALID_OPENAI_MODELS, VALID_YEARS, VALID_MIMETYPES
from main import index, retrieve, load_dotenv


app = FastAPI()


# Load environment variables
load_dotenv()


# Request and Response Models
class IndexRequest(Config):
    pass

class RetrieveRequest(Config):
    pass

class RetrieveResponse(BaseModel):
    response: str
    reference_files: str


@app.post("/index")
async def index_documents(request: IndexRequest):
    """
    Endpoint to index documents from Google Drive.
    """
    try:
        # Pass the validated Pydantic model directly to the index function
        index(request)
        return {"message": "Indexing completed successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/retrieve", response_model=RetrieveResponse)
async def retrieve_documents(request: RetrieveRequest):
    """
    Endpoint to retrieve documents based on a query.
    """
    try:
        # Pass the validated Pydantic model directly to the retrieve function
        response, files_list = retrieve(request)
        reference_files = [f"""- Document {i} : "{file.get("name")}" from year {file.get("year")}"""for i, file in enumerate(files_list, start=1)]
        reference_files = "\n".join(reference_files)
        return {"response": response, "reference_files": reference_files}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
async def root():
    """
    Root endpoint to check the API status.
    """
    return {"message": "Aaltoes RAG API is running"}


