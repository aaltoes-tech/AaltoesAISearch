from pydantic import BaseModel, field_validator, model_validator
from typing import Literal, Optional, Union, List, Dict, Any
import os

VALID_MIMETYPES = [
    "application/vnd.google-apps.document",
    "application/vnd.google-apps.presentation",
    # "application/vnd.google-apps.spreadsheet",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    # "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "application/pdf"
]
FILE_TYPE_MAP = {
    "MSWords": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "MSPP": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    "MSExcel": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "PDF": "application/pdf"
}
VALID_YEARS = [
    "2024",
    "2023",
    "2022",
    "2021",
    "2020",
]  # Add more years as needed
VALID_GEMINI_MODELS = []#["gemini-2.0-flash"]
VALID_OPENAI_MODELS = ["gpt-4o"]

# Set up concurrency controls
class PrfConfig(BaseModel):
    """
    Configuration for concurrency settings.
    """
    max_num_cores:int = 8  # Adjust based on your system CPU
    max_concurrent_indexing:int = 16   # Adjust based on system memory and CPU
    
    @field_validator("max_num_cores")
    def validate_max_num_cores(cls, value):
        # the value must be between 0 and the number of cores
        if value < 0 or value > os.cpu_count():
            raise ValueError(f"Invalid number of cores. Choose between 0 and {os.cpu_count() - 1}.")
        return value
    
    @model_validator(mode="after")
    def validate_model(cls, values):
        # Ensure max_num_cores is less than max_concurrent_indexing
        if values.max_num_cores >= values.max_concurrent_indexing:
            raise ValueError(f"Invalid number of cores. Choose less than {cls.max_concurrent_indexing}.")
        return values


class APIConfig(BaseModel):
    mode: str
    model: str
    emb_func: str
    indx_years: Union[List[str], str] = "Full"
    query: str = "What was the purpose of Aaltoes?"
    top_k: int = 5
    retr_year: Union[int, str] = "Full"
    file_type: str = "Full",
    resume_checkpoint: Optional[bool] = False

    @field_validator("mode")
    def validate_mode(cls, value):
        if value not in ["index", "retrieve"]:
            raise ValueError("Invalid mode. Choose 'index' or 'retrieve'.")
        return value

    @field_validator("model")
    def validate_model(cls, value):
        if value not in VALID_GEMINI_MODELS + VALID_OPENAI_MODELS:
            raise ValueError(f"Invalid model: {value}")
        return value

    @field_validator("emb_func")
    def validate_emb_func(cls, value):
        if value not in ["google", "openai"]:
            raise ValueError(f"Invalid embedding function: {value}")
        return value

    @field_validator("indx_years", mode="before")
    def validate_years(cls, value):
        if value != "Full" and not all(year in VALID_YEARS for year in value):
            raise ValueError(f"Invalid years for indexing: {value}")
        return value

    @field_validator("retr_year", mode="before")
    def validate_retr_year(cls, value):
        if value != "Full" and value not in VALID_YEARS:
            raise ValueError(f"Invalid year for retrieval: {value}")
        return value

    @field_validator("file_type")
    def validate_file_type(cls, value):
        if value not in ["Full", "MSWords", "MSPP", "MSExcel", "PDF"]:
            raise ValueError(f"Invalid file type: {value}")
        return value
    
    @model_validator(mode="after")
    def validate_resume_checkpoint(cls, values):
        # To be only used with mode == "index"
        if values.mode == "retrieve" and values.resume_checkpoint:
            raise ValueError("resume_checkpoint can only be used with mode 'index'.")
        return values