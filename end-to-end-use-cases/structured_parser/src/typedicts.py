"""
Type definitions for structured document extraction.

This module provides TypedDict classes for type safety and better IDE support
when working with document extraction data structures.
"""

from typing import Any, Dict, List, Optional, TypedDict, Union

from vllm import SamplingParams


class MessageContent(TypedDict):
    """
    Type definition for message content in LLM requests.

    Supports both text and image content types for multimodal LLM interactions.
    """

    type: str  # Content type: "text" or "image_url"
    text: Optional[str]  # Text content for text type
    image_url: Optional[Dict[str, str]]  # Image URL data for image_url type


class Message(TypedDict):
    """
    Type definition for a message in an LLM inference request.

    Represents a single message in a conversation with role and content.
    """

    role: str  # Message role: "system", "user", or "assistant"
    content: Union[
        str, List[MessageContent]
    ]  # Message content as string or multimodal list


class InferenceRequest(TypedDict, total=False):
    """
    Type definition for LLM inference request parameters.

    Contains all parameters needed for LLM inference including model settings,
    messages, and generation parameters.
    """

    model: str  # Model identifier
    messages: List[Message]  # Conversation messages
    temperature: float  # Sampling temperature (0.0-1.0)
    top_p: float  # Nucleus sampling parameter
    max_completion_tokens: int  # Maximum tokens to generate
    seed: int  # Random seed for reproducibility
    response_format: Optional[Dict[str, Any]]  # Optional structured output format


class VLLMInferenceRequest(TypedDict):
    """
    Type definition for VLLM inference request format.

    Batch format specifically for VLLM engine processing multiple requests
    with corresponding sampling parameters.
    """

    messages: List[List[Message]]  # Batch of message sequences
    sampling_params: Union[
        SamplingParams, List[SamplingParams]
    ]  # VLLM sampling parameters


class TextArtifact(TypedDict):
    """
    Type definition for extracted text artifacts.

    Represents text content extracted from documents with optional metadata.
    """

    content: str  # Main text content
    notes: Optional[str]  # Additional notes or observations about the text


class ImageArtifact(TypedDict, total=False):
    """
    Type definition for extracted image artifacts.

    Represents images, charts, and visual elements extracted from documents
    with positional and descriptive metadata.
    """

    description: str  # Detailed description of the image
    caption: str  # Caption or label associated with the image
    image_type: str  # Type of image (e.g., 'photograph', 'chart', 'diagram')
    position_top: Optional[str]  # Approximate vertical position
    position_left: Optional[str]  # Approximate horizontal position


class ChartArtifact(TypedDict, total=False):
    """
    Type definition for extracted chart artifacts.

    Represents charts and graphs with structured data and metadata.
    """

    chart_type: str  # Type of chart (e.g., 'bar', 'line', 'pie')
    description: str  # Detailed description of the chart
    caption: str  # Caption or title of the chart
    data: Dict[str, Any]  # Structured chart data


class TableArtifact(TypedDict, total=False):
    """
    Type definition for extracted table artifacts.

    Represents tabular data with structured contents and descriptive information.
    """

    table_contents: Dict[str, Any]  # Structured table data
    table_info: str  # Descriptive information about the table


class ArtifactCollection(TypedDict, total=False):
    """
    Type definition for a collection of extracted artifacts from a document page.

    Groups all types of artifacts extracted from a single document page.
    """

    text: List[TextArtifact]  # Text artifacts from the page
    images: List[ImageArtifact]  # Image artifacts from the page
    tables: List[TableArtifact]  # Table artifacts from the page
    charts: List[ChartArtifact]  # Chart artifacts from the page


class ExtractedPage(TypedDict):
    """
    Type definition for a complete extracted document page.

    Represents a single page from a document with its metadata and all
    extracted artifacts.
    """

    doc_path: str  # Path to the source document
    image_path: str  # Path to the page image file
    page_num: int  # Page number (0-indexed)
    artifacts: ArtifactCollection  # All artifacts extracted from this page
