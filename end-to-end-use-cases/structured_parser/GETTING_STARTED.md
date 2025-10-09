# Getting Started with Structured Document Parser

This guide walks you through setting up and using the Structured Document Parser tool to extract text, tables, and images from PDF documents.

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure the Tool

Edit the `src/config.yaml` file to configure the tool:

```yaml
# Choose your inference backend
model:
  backend: openai-compat  # Use "offline-vllm" for local inference

  # If using openai-compat
  base_url: "https://api.llama.com/compat/v1"
  api_key: "YOUR_API_KEY"
  model_id: "Llama-4-Maverick-17B-128E-Instruct-FP8"  # Or your preferred model
```

## Basic Usage Examples

### Extract Text from a PDF

```bash
python src/structured_extraction.py path/to/document.pdf --text
```

This will:
1. Convert each PDF page to an image
2. Run LLM inference to extract text
3. Save extracted text as JSON in the `extracted` directory

### Extract Text and Tables

```bash
python src/structured_extraction.py path/to/document.pdf --text --tables
```

### Extract All Types of Content

```bash
python src/structured_extraction.py path/to/document.pdf --text --tables --images
```

### Process Multiple PDFs

```bash
python src/structured_extraction.py path/to/pdf_directory --text --tables
```

## Working with Extraction Results

### Export Tables to CSV

```bash
python src/structured_extraction.py path/to/document.pdf --tables --save_tables_as_csv
```

Tables will be saved as individual CSV files in `extracted/tables_TIMESTAMP/`.

### Export Tables to Excel

```bash
python src/structured_extraction.py path/to/document.pdf --tables --export_excel
```

Tables will be combined into a single Excel file with multiple sheets.

### Save to Database

```bash
python src/structured_extraction.py path/to/document.pdf --text --tables --save_to_db
```

Extracted content will be stored in an SQLite database for structured querying.

## Python API Examples

### Extract Content Programmatically

```python
from src.structured_extraction import ArtifactExtractor
from src.utils import PDFUtils

# Extract pages from a PDF
pdf_pages = PDFUtils.extract_pages("document.pdf")

# Process each page
for page in pdf_pages:
    # Extract text
    text_artifacts = ArtifactExtractor.from_image(
        page["image_path"], ["text"]
    )

    # Or extract multiple artifact types
    all_artifacts = ArtifactExtractor.from_image(
        page["image_path"], ["text", "tables", "images"]
    )

    # Process the extracted artifacts
    print(all_artifacts)
```

### Query the Database

```python
from src.json_to_sql import DatabaseManager

# Query all text artifacts
text_df = DatabaseManager.sql_query(
    "sqlite3.db",
    "SELECT * FROM document_artifacts WHERE artifact_type = 'text'"
)

# Query tables containing specific content
revenue_tables = DatabaseManager.sql_query(
    "sqlite3.db",
    "SELECT * FROM document_artifacts WHERE artifact_type = 'table' AND table_info LIKE '%revenue%'"
)
```

### Semantic Search

```python
from src.json_to_sql import VectorIndexManager

# Search for relevant content
results = VectorIndexManager.knn_query(
    "What is the revenue growth for Q2?",
    "chroma.db",
    n_results=5
)

# Display results
for i, (doc_id, distance, content) in enumerate(zip(
    results['ids'], results['distances'], results['documents']
)):
    print(f"Result {i+1} (similarity: {1-distance:.2f}):")
    print(content[:200] + "...\n")
```

## Customizing Extraction

### Modify Prompts

Edit the prompts in `src/config.yaml` to improve extraction for your specific document types:

```yaml
artifacts:
  text:
    prompts:
      system: "You are an OCR expert. Your task is to extract all text sections..."
      user: "TARGET SCHEMA:\n```json\n{schema}\n```"
```

### Add a Custom Artifact Type

1. Add configuration to `src/config.yaml`:

```yaml
artifacts:
  my_custom_type:
    prompts:
      system: "Your custom system prompt..."
      user: "Your custom user prompt with {schema} placeholder..."
    output_schema: {
      # Your schema definition here
    }
    use_json_decoding: true
```

2. Update the CLI in `src/structured_extraction.py`:

```python
def main(
    target_path: str,
    text: bool = True,
    tables: bool = False,
    images: bool = False,
    my_custom_type: bool = False,  # Add your type here
    save_to_db: bool = False,
    ...
):
    # Update artifact types logic
    to_extract = []
    if text:
        to_extract.append("text")
    if tables:
        to_extract.append("tables")
    if images:
        to_extract.append("images")
    if my_custom_type:
        to_extract.append("my_custom_type")  # Add your type here
```

## Troubleshooting

### LLM Response Format Issues

If the LLM responses aren't being correctly parsed, check:
1. Your output schema in `config.yaml`
2. The `use_json_decoding` setting (set to `true` for more reliable parsing)
3. Consider using a larger model or reducing extraction complexity

### Database Issues

If you encounter database errors:
1. Ensure SQLite is properly installed
2. Check database file permissions
3. Use `DatabaseManager.create_artifact_table()` to reinitialize the table schema

### PDF Rendering Issues

If PDF extraction quality is poor:
1. Try adjusting the DPI setting in `PDFUtils.extract_pages()`
2. For complex layouts, split extraction into smaller chunks (per section)
3. Consider pre-processing PDFs with OCR tools for better text layer quality

## Next Steps

- Try extracting from different types of documents
- Adjust prompts and schemas for your specific use cases
- Explore the vector search capabilities for semantic document queries
- Integrate with your existing document processing workflows
