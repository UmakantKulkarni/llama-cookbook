"""
Structured data extraction module for processing images with LLMs.

This module provides functionality to extract structured data from images using
local or API-based LLMs. It handles the preparation of requests, batching for
efficient inference, and parsing of responses into structured formats.
"""

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import fire

from json_to_table import flatten_json_to_sql, json_to_csv
from tqdm import tqdm
from typedicts import ArtifactCollection, ExtractedPage, InferenceRequest

from utils import (
    config,
    export_csvs_to_excel_tabs,
    ImageUtils,
    InferenceUtils,
    JSONUtils,
    PDFUtils,
)


# Constants
EXTRACTED_DATA_KEY = "extracted_data"
SUPPORTED_BACKENDS = ["offline-vllm", "openai-compat"]
SUPPORTED_FILE_TYPES = [".pdf"]


def setup_logger(logfile: str, verbose: bool = False) -> logging.Logger:
    """
    Set up a logger for the application with file and optional console output.

    Args:
        logfile: Path to the log file
        verbose: If True, also log to console

    Returns:
        Configured logger instance
    """
    # Create a logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    # Create a file handler
    file_handler = logging.FileHandler(logfile)
    file_handler.setLevel(logging.DEBUG)

    # Create a formatter and set it for the file handler
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(formatter)

    # Add the file handler to the logger
    logger.addHandler(file_handler)

    # If verbose, also add a console handler
    if verbose:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger


logger = setup_logger("app.log", verbose=False)


class RequestBuilder:
    """Builder for LLM inference requests."""

    @staticmethod
    def build(
        img_path: str,
        system_prompt: str,
        user_prompt: str,
        output_schema: Dict[str, Any],
        use_json_decoding: bool = False,
        model: Optional[str] = None,
    ) -> InferenceRequest:
        """
        Build an inference request for an image.

        Args:
            img_path: Path to the image file
            system_prompt: System prompt for the LLM
            user_prompt: User prompt for the LLM
            output_schema: JSON schema for the output
            use_json_decoding: Whether to use JSON-guided decoding
            model: Optional model override

        Returns:
            InferenceRequest: Formatted request for the LLM

        Raises:
            FileNotFoundError: If the image file doesn't exist
        """
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image file not found: {img_path}")

        img_b64 = ImageUtils.encode_image(img_path)

        # Create a copy of the inference config to avoid modifying the original
        request_params = dict(config["extraction_inference"])
        request_params["messages"] = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{img_b64}"},
                    },
                    {"type": "text", "text": user_prompt},
                ],
            },
        ]

        if use_json_decoding:
            request_params["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "OutputSchema", "schema": output_schema},
            }

        if model:
            request_params["model"] = model

        return request_params


class ArtifactExtractor:
    """Extractor for document artifacts."""

    @staticmethod
    def _prepare_inference_requests(
        img_path: str, artifact_types: List[str]
    ) -> List[Tuple[str, InferenceRequest]]:
        """
        Prepare inference requests for each artifact type.

        Args:
            img_path: Path to the image file
            artifact_types: Types of artifacts to extract

        Returns:
            List of tuples containing (artifact_type, inference_request)
        """
        requests = []
        for artifact in artifact_types:
            art_config = config["artifacts"].get(artifact)
            if not art_config:
                logger.warning(f"No configuration found for artifact type: {artifact}")
                continue

            system_prompt = art_config["prompts"].get("system", "")
            user_prompt = art_config["prompts"].get("user", "")
            output_schema = art_config.get("output_schema", None)
            use_json_decoding = art_config.get("use_json_decoding", False)

            if user_prompt and output_schema is not None:
                user_prompt = user_prompt.format(schema=json.dumps(output_schema))

            request = RequestBuilder.build(
                img_path,
                system_prompt,
                user_prompt,
                output_schema,
                use_json_decoding,
            )
            requests.append((artifact, request))

        return requests

    @staticmethod
    def _run_inference(
        requests: List[Tuple[str, InferenceRequest]],
    ) -> List[Tuple[str, str]]:
        """
        Run inference for all requests.

        Args:
            requests: List of tuples containing (artifact_type, inference_request)

        Returns:
            List of tuples containing (artifact_type, response)

        Raises:
            ValueError: If the backend is not supported
        """
        backend = config["model"].get("backend")
        if backend not in SUPPORTED_BACKENDS:
            raise ValueError(
                f"Allowed config.model.backend: {SUPPORTED_BACKENDS}, got unknown value: {backend}"
            )

        artifact_types = [r[0] for r in requests]
        inference_requests = [r[1] for r in requests]

        response_batch = []
        if backend == "offline-vllm":
            request_batch = InferenceUtils.make_vllm_batch(inference_requests)
            response_batch = InferenceUtils.run_vllm_inference(request_batch)
        elif backend == "openai-compat":
            response_batch = [
                InferenceUtils.run_openai_inference(request)
                for request in inference_requests
            ]

        return list(zip(artifact_types, response_batch))

    @staticmethod
    def _process_responses(responses: List[Tuple[str, str]]) -> ArtifactCollection:
        """
        Process responses into a structured artifact collection.

        Args:
            responses: List of tuples containing (artifact_type, response)

        Returns:
            ArtifactCollection: Extracted artifacts
        """
        extracted = {}
        for artifact_type, raw_response in responses:
            try:
                json_response = JSONUtils.extract_json_from_response(raw_response)

                if EXTRACTED_DATA_KEY in json_response:
                    json_response = json_response[EXTRACTED_DATA_KEY]

                extracted.update(json_response)
            except Exception as e:
                logger.error(f"Failed to process response for {artifact_type}: {e}")
                extracted.update({artifact_type: {"error": str(e)}})

        return extracted

    @staticmethod
    def from_image(
        img_path: str,
        artifact_types: Union[List[str], str],
    ) -> ArtifactCollection:
        """
        Extract artifacts from an image.

        Args:
            img_path: Path to the image file
            artifact_types: Type(s) of artifacts to extract

        Returns:
            ArtifactCollection: Extracted artifacts

        Raises:
            ValueError: If the backend is not supported
            FileNotFoundError: If the image file doesn't exist
        """
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image file not found: {img_path}")

        if isinstance(artifact_types, str):
            artifact_types = [artifact_types]

        # Prepare inference requests
        requests = ArtifactExtractor._prepare_inference_requests(
            img_path, artifact_types
        )

        # Run inference
        responses = ArtifactExtractor._run_inference(requests)

        # Process responses
        return ArtifactExtractor._process_responses(responses)

    @staticmethod
    def from_pdf(pdf_path: str, artifact_types: List[str]) -> List[ExtractedPage]:
        """
        Extract artifacts from all pages in a PDF.

        Args:
            pdf_path: Path to the PDF file
            artifact_types: Types of artifacts to extract

        Returns:
            List[ExtractedPage]: Extracted pages with artifacts

        Raises:
            FileNotFoundError: If the PDF file doesn't exist
        """
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        pdf_pages = PDFUtils.extract_pages(pdf_path)
        logger.info(f"Processing {len(pdf_pages)} pages from {pdf_path}")
        for page in tqdm(pdf_pages, desc="Processing PDF pages"):
            try:
                page_artifacts = ArtifactExtractor.from_image(
                    page["image_path"], artifact_types
                )
                page_artifacts = json.loads(json.dumps(page_artifacts))
                page["artifacts"] = page_artifacts
            except Exception as e:
                logger.error(
                    f"Error processing page {page['page_num']} in {pdf_path}: {e}"
                )
                page["artifacts"] = {"error": f"Error {e} in artifact extraction"}

        return pdf_pages


def get_target_files(target_path: str) -> List[Path]:
    """
    Get list of files to process.

    Args:
        target_path: Path to a file or directory

    Returns:
        List of Path objects to process

    Raises:
        FileNotFoundError: If the target path doesn't exist
        ValueError: If the file type is unsupported
    """
    if not os.path.exists(target_path):
        raise FileNotFoundError(f"Target path not found: {target_path}")

    target_path_obj = Path(target_path)
    if target_path_obj.is_file() and target_path_obj.suffix not in SUPPORTED_FILE_TYPES:
        raise ValueError(
            f"Unsupported file type: {target_path_obj.suffix}. Supported types: {SUPPORTED_FILE_TYPES}"
        )

    targets = (
        [target_path_obj]
        if target_path_obj.is_file()
        else [f for f in target_path_obj.iterdir() if f.suffix in SUPPORTED_FILE_TYPES]
    )
    logger.debug(f"Processing {len(targets)} files")
    if not targets:
        logger.warning(f"No supported files found in {target_path}")

    return targets


def process_files(
    targets: List[Path], artifact_types: List[str]
) -> List[Dict[str, Any]]:
    """
    Process files and extract artifacts.

    Args:
        targets: List of files to process
        artifact_types: Types of artifacts to extract

    Returns:
        List of extracted artifacts
    """
    out_json = []
    for target in targets:
        try:
            artifacts = ArtifactExtractor.from_pdf(str(target), artifact_types)
            out_json.extend(artifacts)
        except Exception as e:
            logger.error(f"Failed to process {target}: {e}")
    return out_json


def save_results(
    output_dir: Path,
    data: List[Dict[str, Any]],
    save_to_db: bool = False,
    save_tables_as_csv: bool = False,
    export_excel: bool = False,
) -> None:
    """
    Save extraction results to a file and optionally to SQL and vector databases.

    Args:
        output_path: Path to save the JSON results
        data: Data to save
        save_to_sql: Whether to save to SQL database
        sql_db_path: Path to the SQLite database file
        save_to_vector: Whether to save to vector database
        vector_db_path: Path to the vector database
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save to JSON file
    output_path = None
    try:
        output_path = output_dir / f"artifacts_{timestamp}.json"
        json_content = json.dumps(data, indent=2)
        output_path.write_text(json_content)
        logger.info(f"Extracted artifacts written to {output_path}")
    except Exception as e:
        logger.error(f"Failed to write output file: {e}")

    if save_tables_as_csv or export_excel:
        tables_charts = sum([x["artifacts"]["tables"] for x in data], []) + sum(
            [x["artifacts"]["charts"] for x in data], []
        )
        for tab in tables_charts:
            # llm: convert each table to a csv string
            csv_string, filename = json_to_csv(tab)
            outfile = output_dir / f"tables_{timestamp}" / filename
            outfile.parent.mkdir(parents=True, exist_ok=True)
            outfile.write_text(csv_string)
            logger.info(f"Extracted table written to {outfile}")

        if export_excel:
            output_path = output_dir / f"tables_{timestamp}.xlsx"
            export_csvs_to_excel_tabs(output_dir / f"tables_{timestamp}", output_path)

    # Save to SQL and vector databases
    if save_to_db:
        # Get database paths from config
        sql_db_path = config.get("database", {}).get("sql_db_path", None)
        vector_db_path = config.get("database", {}).get("vector_db_path", None)
        assert (
            sql_db_path is not None
        ), "Save to SQL failed; SQL database path not found in config"

        # Save to SQL and optionally to vector database
        counts = flatten_json_to_sql(str(output_path), sql_db_path, vector_db_path)
        logger.info(
            f"Extracted {counts.get('text', 0)} text artifacts, {counts.get('image', 0)} image artifacts, and {counts.get('table', 0)} table artifacts from {len(data)} pages."
        )
        logger.info(f"Extracted artifacts saved to SQL database: {sql_db_path}")
        logger.info(f"Extracted artifacts indexed in vector database: {vector_db_path}")


def main(
    target_path: str,
    artifacts: str,
    save_to_db: bool = False,
    save_tables_as_csv: bool = True,
    export_excel: bool = False,
) -> None:
    """
    Extract structured data from PDF documents using LLM-powered extraction.

    Processes PDFs to extract text, tables, images, and charts as structured JSON.
    Outputs are saved to timestamped files and optionally to databases.

    Args:
        target_path: PDF file or directory path to process
        artifacts: Comma-separated artifact types (e.g. "text,tables,images,charts")
        save_to_db: Save to SQL/vector databases if True
        save_tables_as_csv: Export tables as individual CSV files if True
        export_excel: Combine all tables into single Excel workbook if True

    Output:
        - JSON file with all extracted artifacts
        - CSV files for each table (if save_tables_as_csv=True)
        - Excel workbook with all tables (if export_excel=True)
        - Database records (if save_to_db=True)

    Raises:
        ValueError: Invalid artifact types or unsupported file format
        FileNotFoundError: Target path does not exist
    """
    ALLOWED_ARTIFACTS = list(config["artifacts"].keys())
    artifact_types = [x for x in artifacts if x in ALLOWED_ARTIFACTS]
    print("Extracting artifacts: ", artifact_types, "\n")

    # Get files to process
    targets = get_target_files(target_path)
    if not targets:
        return

    # Process files
    results = process_files(targets, artifact_types)

    # Save results
    target_path_obj = Path(target_path)
    output_dir = target_path_obj.parent / "extracted"
    save_results(
        output_dir,
        results,
        save_to_db=save_to_db,
        save_tables_as_csv=save_tables_as_csv,
        export_excel=export_excel,
    )


if __name__ == "__main__":
    fire.Fire(main)
