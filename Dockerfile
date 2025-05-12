# Use Python 3.12 slim as base image
FROM python:3.12-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DEBIAN_FRONTEND=noninteractive

# Install system dependencies required for unstructured[all-docs]
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    libmagic1 \
    libpq-dev \
    pkg-config \
    poppler-utils \
    tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# Set working directory
WORKDIR /app

# Copy project files
COPY pyproject.toml uv.lock ./
COPY . .

# Install Python dependencies using uv
RUN uv pip install --system .

# Create directory for ChromaDB persistence
RUN mkdir -p /app/chroma_db

# Expose port for FastAPI
EXPOSE 8000

# Command to run the FastAPI service
CMD ["uvicorn", "fapi_app:app", "--host", "0.0.0.0", "--port", "8000"] 