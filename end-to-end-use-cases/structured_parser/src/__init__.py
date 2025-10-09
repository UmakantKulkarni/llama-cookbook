"""
Structured Parser Package

This package provides tools for extracting structured data from documents,
particularly PDFs, using LLMs. It includes functionality for:

1. Extracting text, tables, and images from PDFs
2. Converting extracted data to SQL database entries
3. Creating vector embeddings for semantic search
"""

from .json_to_sql import DatabaseManager, flatten_json_to_sql, VectorIndexManager
from .structured_extraction import (
    ArtifactExtractor,
    main as extract_artifacts,
    RequestBuilder,
)
from .utils import config, ImageUtils, InferenceUtils, JSONUtils, load_config, PDFUtils

__all__ = [
    # Main extraction functionality
    "ArtifactExtractor",
    "RequestBuilder",
    "extract_artifacts",
    # Database functionality
    "DatabaseManager",
    "VectorIndexManager",
    "flatten_json_to_sql",
    "sql_query",
    # Utility classes
    "ImageUtils",
    "JSONUtils",
    "PDFUtils",
    "InferenceUtils",
    # Configuration
    "config",
    "load_config",
]
