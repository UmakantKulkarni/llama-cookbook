# DocumentLens: Rich Document Parsing with LLMs

A powerful, LLM-based tool for extracting structured data from rich documents (PDFs) with Llama models.

## Overview

This tool uses Llama models to extract text, tables, images, and charts from PDFs, converting unstructured document data into structured, machine-readable formats. It supports:

- **Text extraction**: Extract and structure main text, titles, captions, etc.
- **Table extraction**: Convert complex tables into structured data formats
- **Image extraction**: Extract images with contextual descriptions and captions
- **Chart extraction**: Convert charts and graphs into structured JSON data
- **Multiple output formats**: JSON, CSV, Excel, and SQL database storage
- **Vector search capabilities**: Semantic search across extracted content

The tool is designed to handle complex documents with high accuracy and provides flexible configuration options to tailor extraction tasks to specific needs.


## Project Structure

```
structured_parser/
├── src/
│   ├── structured_extraction.py  # Main entry point and extraction logic
│   ├── utils.py                  # Utility functions and classes
│   ├── typedicts.py             # Type definitions
│   ├── json_to_table.py         # Database integration functions
│   └── config.yaml              # Configuration file
├── pdfs/                        # Sample PDFs and extraction results
├── README.md                    # This file
├── CONTRIBUTING.md              # Contribution guidelines
└── requirements.txt             # Python dependencies
```


## Installation

### Prerequisites

- Python 3.9+
- [Optional] Local GPU for offline inference

### Setup

1. Clone the repository

```bash
git clone https://github.com/meta-llama/llama-cookbook.git
cd llama-cookbook
```

2. Install project specific dependencies:
```bash
cd end-to-end-use-cases/structured_parser
pip install -r requirements.txt
```

### Configure the tool (see [Configuration](#Configuration) section)
(Note: Setup API Key, Model for inferencing, etc.)

### Extract text from a PDF:

```bash
python src/structured_extraction.py path/to/document.pdf text
```

### Extract charts and tables, and save them as CSV files:

```bash
python src/structured_extraction.py path/to/document.pdf charts,tables --save_tables_as_csv
```

### Process a directory of PDFs and export tables to Excel:

```bash
python src/structured_extraction.py path/to/pdf_directory text,tables --export_excel
```

### Extract all artifact types and save to database and as Excel sheets:

```bash
python src/structured_extraction.py path/to/document.pdf text,tables,images,charts --save_to_db --export_excel
```

## Configuration

The tool is configured via `src/config.yaml`. Key configuration options include:

### Model Configuration

```yaml
model:
  backend: openai-compat  # [offline-vllm, openai-compat]

  # For openai-compat
  base_url: "https://api.llama.com/compat/v1"
  api_key: "YOUR_API_KEY"
  model_id: "Llama-4-Maverick-17B-128E-Instruct-FP8"

  # For offline-vllm
  path: "/path/to/checkpoint"
  tensor_parallel_size: 4
  max_model_len: 32000
  max_num_seqs: 32
```

### Inference Parameters

```yaml
extraction_inference:
  temperature: 0.2
  top_p: 0.9
  max_completion_tokens: 32000
  seed: 42
```

### Database Configuration

```yaml
database:
  sql_db_path: "sqlite3.db"
  vector_db_path: "chroma.db"
```

### Artifact Configuration

The tool includes configurable prompts and output schemas for each artifact type (text, tables, images, charts). These can be modified in the `config.yaml` file to customize extraction behavior for specific document types.

## Output Formats

### JSON Output
The primary output format includes all extracted artifacts in a structured JSON format with timestamps.

### CSV Export
Tables and charts can be exported as individual CSV files for easy analysis in spreadsheet applications.

### Excel Export
Multiple tables can be combined into a single Excel workbook with separate tabs for each table.

### Database Storage
Extracted data can be stored in SQLite databases with optional vector indexing for semantic search.

## API Usage

### Programmatic Usage

```python
from src.structured_extraction import ArtifactExtractor
from src.utils import PDFUtils

# Extract pages from PDF
pages = PDFUtils.extract_pages("document.pdf")

# Process specific pages
for page in pages[10:20]:  # Process pages 10-19
    artifacts = ArtifactExtractor.from_image(
        page["image_path"],
        ["text", "tables"]
    )
    # Custom processing of artifacts...
```

### Single Image Processing

```python
from src.structured_extraction import ArtifactExtractor

# Extract from a single image
artifacts = ArtifactExtractor.from_image(
    "path/to/image.png",
    ["text", "tables", "images"]
)
```

## Architecture

### Core Components

1. **RequestBuilder**: Builds inference requests for LLMs with image and text content
2. **ArtifactExtractor**: Extracts structured data from documents using configurable prompts
3. **PDFUtils**: Handles PDF processing and page extraction as images
4. **InferenceUtils**: Manages LLM inference with support for VLLM and OpenAI-compatible APIs
5. **JSONUtils**: Handles JSON extraction and validation from LLM responses
6. **ImageUtils**: Utility functions for image encoding and processing

### Data Flow

1. PDFs are converted to images (one per page) using PyMuPDF
2. Images are processed by the LLM to extract structured data based on configured prompts
3. Structured data is saved in various formats (JSON, CSV, SQL, etc.)
4. Optional vector indexing for semantic search capabilities

### Supported Artifact Types

- **text**: Main text content, titles, captions, and other textual elements
- **tables**: Structured tabular data with proper formatting
- **images**: Image descriptions, captions, and metadata
- **charts**: Chart data extraction with structured format including axes, data points, and metadata

## Extending the Tool

### Adding New Artifact Types

1. Add a new artifact type configuration in `config.yaml`:

```yaml
artifacts:
  my_new_artifact:
    prompts:
      system: "Your system prompt here..."
      user: "Your user prompt with {schema} placeholder..."
    output_schema: {
      # Your JSON schema here
    }
    use_json_decoding: true
```



### Customizing Extraction Logic

The extraction logic is modular and can be customized by:

1. Modifying prompts in the `config.yaml` file
2. Adjusting output schemas to capture different data structures
3. Extending the `ArtifactExtractor` class for specialized extraction needs

### Using Different Models

The tool supports two backends:

1. **openai-compat**: Any API compatible with the OpenAI API format (including Llama API)
2. **offline-vllm**: Local inference using VLLM for self-hosted deployments

## Best Practices

1. **Model Selection**: Use larger models for complex documents or when high accuracy is required
2. **Prompt Engineering**: Adjust prompts in `config.yaml` for your specific document types
3. **Output Schema**: Define precise schemas to guide the model's extraction process


## Troubleshooting

### Common Issues

- **Model capacity errors**: Reduce max tokens or use a larger model
- **Extraction quality issues**: Adjust prompts or output schemas
- **Configuration errors**: Verify model paths and API credentials in config.yaml
