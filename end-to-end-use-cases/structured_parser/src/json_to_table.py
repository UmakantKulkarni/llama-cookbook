"""
JSON to SQL conversion module for structured document data.

This module provides functionality to convert structured JSON data extracted from
documents into SQLite database entries and vector embeddings for semantic search.
"""

import json
import logging
import os
import sqlite3
from typing import Any, Dict, List, Optional, Tuple

import chromadb
import pandas as pd
from chromadb.config import Settings

from utils import config, InferenceUtils, JSONUtils


def setup_logger(logfile, verbose=False):
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


logger = logging.getLogger(__name__)


class DatabaseManager:
    """Manager for database operations."""

    @staticmethod
    def validate_path(path: str) -> str:
        """
        Validate that a file path exists or can be created.

        Args:
            path: Path to validate

        Returns:
            Validated path

        Raises:
            ValueError: If path is invalid
        """
        if not path:
            raise ValueError("Database path cannot be empty")

        # Ensure directory exists
        directory = os.path.dirname(path)
        if directory and not os.path.exists(directory):
            try:
                os.makedirs(directory, exist_ok=True)
            except OSError as e:
                raise ValueError(f"Cannot create directory for database: {e}")

        return path

    @staticmethod
    def create_artifact_table(sql_db_path: str) -> None:
        """
        Create the SQL table schema for storing document artifacts.

        Args:
            sql_db_path: Path to the SQLite database file

        Raises:
            sqlite3.Error: If there's an error creating the table
        """
        sql_db_path = DatabaseManager.validate_path(sql_db_path)

        try:
            with sqlite3.connect(sql_db_path) as conn:
                cursor = conn.cursor()

                # Drop table if it exists
                cursor.execute("DROP TABLE IF EXISTS document_artifacts")

                # Create table with schema
                cursor.execute("""
                CREATE TABLE IF NOT EXISTS document_artifacts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    doc_path TEXT,
                    page_num INTEGER,
                    artifact_type TEXT,  -- 'table', 'text', or 'image'

                    -- Common metadata
                    content_json TEXT,   -- JSON string of the artifact content

                    -- Table specific fields
                    table_info TEXT,

                    -- Text specific fields
                    text_content TEXT,
                    text_notes TEXT,

                    -- Image specific fields
                    image_position_top TEXT,
                    image_position_left TEXT,
                    image_description TEXT,
                    image_caption TEXT,
                    image_type TEXT
                )
                """)

                # Create indexes for common queries
                cursor.execute(
                    "CREATE INDEX IF NOT EXISTS idx_artifact_type ON document_artifacts(artifact_type)"
                )
                cursor.execute(
                    "CREATE INDEX IF NOT EXISTS idx_doc_path ON document_artifacts(doc_path)"
                )

                conn.commit()
        except sqlite3.Error as e:
            print(f"Database error creating table: {e}")
            raise

    @staticmethod
    def sql_query(db_path: str, query: str) -> pd.DataFrame:
        """
        Query the document artifacts table and return results as a DataFrame.

        Args:
            db_path: Path to the SQLite database file
            query: SQL query to execute

        Returns:
            DataFrame containing query results

        Raises:
            ValueError: If query is empty or invalid
            sqlite3.Error: If there's a database error
        """
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")

        if not os.path.exists(db_path):
            raise FileNotFoundError(f"Database file not found: {db_path}")

        try:
            with sqlite3.connect(db_path) as conn:
                df = pd.read_sql_query(query, conn)
            return df
        except (sqlite3.Error, pd.io.sql.DatabaseError) as e:
            print(f"Query error: {e}")
            raise

    @staticmethod
    def export_db(db_path: str, export_path: str) -> None:
        """
        Export the document artifacts table to a CSV file.

        Args:
            db_path: Path to the SQLite database file
            export_path: Path to the CSV file to export

        Raises:
            ValueError: If export path is invalid
            sqlite3.Error: If there's a database error
        """
        if not export_path or not export_path.strip():
            raise ValueError("Export path cannot be empty")

        if not os.path.exists(db_path):
            raise FileNotFoundError(f"Database file not found: {db_path}")

        df = DatabaseManager.sql_query(db_path, "SELECT * FROM document_artifacts")
        df.to_csv(export_path, index=False)


class VectorIndexManager:
    """Manager for vector index operations."""

    @staticmethod
    def write_to_index(
        vector_db_path: str, document_ids: List[str], document_contents: List[str]
    ) -> None:
        """
        Write document contents to a vector index for semantic search.

        Args:
            vector_db_path: Path to the vector database
            document_ids: List of document IDs
            document_contents: List of document contents to index

        Raises:
            ValueError: If inputs are invalid
            RuntimeError: If there's an error writing to the index
        """
        if not document_ids or not document_contents:
            print("No documents to index")
            return

        if len(document_ids) != len(document_contents):
            raise ValueError(
                "document_ids and document_contents must have the same length"
            )

        try:
            client = chromadb.PersistentClient(
                path=vector_db_path, settings=Settings(anonymized_telemetry=False)
            )
            collection = client.get_or_create_collection(name="structured_parser")
            collection.add(
                documents=document_contents,
                ids=document_ids,
            )
            logger.info(f"Added {len(document_ids)} documents to vector index")
        except Exception as e:
            print(f"Error writing to vector index: {e}")
            raise RuntimeError(f"Failed to write to vector index: {e}")

    @staticmethod
    def knn_query(
        query_text: str, vector_db_path: str, n_results: int = 10
    ) -> pd.DataFrame:
        """
        Perform a semantic search query on the vector index.

        Args:
            query_text: Text to search for
            vector_db_path: Path to the vector database
            n_results: Number of results to return

        Returns:
            DataFrame containing query results

        Raises:
            ValueError: If query is empty
            FileNotFoundError: If vector database doesn't exist
            RuntimeError: If there's an error querying the index
        """
        if not query_text or not query_text.strip():
            raise ValueError("Query text cannot be empty")

        if not os.path.exists(vector_db_path):
            raise FileNotFoundError(f"Vector database not found: {vector_db_path}")

        try:
            client = chromadb.PersistentClient(
                path=vector_db_path, settings=Settings(anonymized_telemetry=False)
            )
            collection = client.get_collection(name="structured_parser")

            results = collection.query(query_texts=[query_text], n_results=n_results)
            df = pd.DataFrame(
                {k: results[k][0] for k in ["ids", "distances", "documents"]}
            )
            df["ids"] = df["ids"].apply(int)
            return df
        except Exception as e:
            print(f"Vector query error: {e}")
            raise RuntimeError(f"Failed to query vector index: {e}")


class SQLProcessor:
    """Processor for document data."""

    @staticmethod
    def process_text_artifact(
        cursor: sqlite3.Cursor, doc_path: str, page_num: int, text: Dict[str, Any]
    ) -> Optional[Tuple[str, str]]:
        """
        Process a text artifact and insert it into the database.

        Args:
            cursor: Database cursor
            doc_path: Document path
            page_num: Page number
            text: Text artifact data

        Returns:
            Tuple of (document_id, indexable_content) or None if no content
        """
        if not text or not text.get("content"):
            return None

        cursor.execute(
            """
            INSERT INTO document_artifacts (
                doc_path, page_num, artifact_type,
                text_content, text_notes
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (
                doc_path,
                page_num,
                "text",
                text.get("content", ""),
                text.get("notes", ""),
            ),
        )

        # Prepare for vector indexing
        indexable_content = text.get("content", "") + " | " + text.get("notes", "")
        return str(cursor.lastrowid), indexable_content

    @staticmethod
    def process_image_artifact(
        cursor: sqlite3.Cursor, doc_path: str, page_num: int, image: Dict[str, Any]
    ) -> Tuple[str, str]:
        """
        Process an image artifact and insert it into the database.

        Args:
            cursor: Database cursor
            doc_path: Document path
            page_num: Page number
            image: Image artifact data

        Returns:
            Tuple of (document_id, indexable_content)
        """
        # Skip empty tables
        if not isinstance(image, dict):
            return None

        cursor.execute(
            """
            INSERT INTO document_artifacts (
                doc_path, page_num, artifact_type,
                image_position_top, image_position_left,
                image_description, image_caption, image_type
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                doc_path,
                page_num,
                "image",
                image.get("position_top", ""),
                image.get("position_left", ""),
                image.get("description", ""),
                image.get("caption", ""),
                image.get("image_type", ""),
            ),
        )

        # Prepare for vector indexing
        indexable_content = (
            image.get("image_type", "")
            + " | "
            + image.get("description", "")
            + " | "
            + image.get("caption", "")
        )
        return str(cursor.lastrowid), indexable_content

    @staticmethod
    def process_table_artifact(
        cursor: sqlite3.Cursor, doc_path: str, page_num: int, table: Dict[str, Any]
    ) -> Optional[Tuple[str, str]]:
        """
        Process a table artifact and insert it into the database.

        Args:
            cursor: Database cursor
            doc_path: Document path
            page_num: Page number
            table: Table artifact data

        Returns:
            Tuple of (document_id, indexable_content) or None if empty table
        """
        # Skip empty tables
        if not isinstance(table, dict):
            return None
        if not table.get("table_contents") and not table.get("table_info"):
            return None

        cursor.execute(
            """
            INSERT INTO document_artifacts (
                doc_path, page_num, artifact_type,
                content_json, table_info
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (
                doc_path,
                page_num,
                "table",
                json.dumps(table.get("table_contents", {})),
                table.get("table_info", ""),
            ),
        )

        # Prepare for vector indexing
        indexable_content = f"""{table.get("table_info", "")}\n\n```json {json.dumps(table.get("table_contents", {}))}```
        """
        return str(cursor.lastrowid), indexable_content


def flatten_json_to_sql(
    json_path: str, sql_db_path: str, vector_db_path: Optional[str] = None
) -> None:
    """
    Convert structured JSON data to SQL database entries and optionally index in a vector database.

    Args:
        json_path: Path to the JSON file
        sql_db_path: Path to the SQLite database
        vector_db_path: Optional path to the vector database

    Raises:
        FileNotFoundError: If JSON file doesn't exist
        ValueError: If paths are invalid
        sqlite3.Error: If there's a database error
    """
    # Validate inputs
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"JSON file not found: {json_path}")

    # Create the SQL table if it doesn't exist
    DatabaseManager.create_artifact_table(sql_db_path)

    # Initialize buffers for vector indexing
    document_ids = []
    document_contents = []

    # Counts for logging
    counts = {}

    # Load JSON data
    try:
        with open(json_path, "r") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"Invalid JSON file: {e}")
        raise ValueError(f"Invalid JSON file: {e}")

    # Connect to the database
    try:
        with sqlite3.connect(sql_db_path) as conn:
            cursor = conn.cursor()

            # Process each page in the document
            for page in data:
                doc_path = page.get("doc_path", "")
                page_num = page.get("page_num", 0)
                artifacts = page.get("artifacts", {})

                # Process text
                for text in artifacts.get("text", []):
                    result = SQLProcessor.process_text_artifact(
                        cursor, doc_path, page_num, text
                    )
                    if result:
                        document_ids.append(result[0])
                        document_contents.append(result[1])
                        counts["text"] = counts.get("text", 0) + 1

                # Process images
                for image in artifacts.get("images", []):
                    result = SQLProcessor.process_image_artifact(
                        cursor, doc_path, page_num, image
                    )
                    if result:
                        document_ids.append(result[0])
                        document_contents.append(result[1])
                        counts["image"] = counts.get("image", 0) + 1

                # Process tables
                for table in artifacts.get("tables", []):
                    result = SQLProcessor.process_table_artifact(
                        cursor, doc_path, page_num, table
                    )
                    if result:
                        document_ids.append(result[0])
                        document_contents.append(result[1])
                        counts["table"] = counts.get("table", 0) + 1

            conn.commit()
    except sqlite3.Error as e:
        print(f"Database error: {e}")
        raise

    # Write to vector index
    if vector_db_path and document_ids:
        VectorIndexManager.write_to_index(
            vector_db_path, document_ids, document_contents
        )

    return counts


def json_to_csv(data: dict, info: str = "") -> Tuple[str, str]:
    system_prompt = """You are an expert at converting JSON data to flat csv tables.

You will receive 2 inputs:
1. JSON-formatted data of a table
2. A string describing the contents of the table.

I require 2 things from you:
1. A CSV string representation of the table
2. A succinct filename for this table based on the data contents.

You should only respond with a JSON, no preamble required. Your JSON response should follow this format:
{"csv_table": <str of table>, "filename": <filename to save table>}. Your CSV string should be for a single table that can be loaded into Pandas."""

    user_prompt = f"data:\n{json.dumps(data)}"
    if info:
        user_prompt += f"\n\ninfo:\n{info}"

    request = InferenceUtils.request_builder(
        user_prompt=user_prompt,
        system_prompt=system_prompt,
        temperature=0.4,
        max_completion_tokens=2048,
        top_p=0.9,
        seed=42,
    )
    backend = config["model"].get("backend")
    if backend == "offline-vllm":
        vllm_request_batch = InferenceUtils.make_vllm_batch(request)
        raw_response = InferenceUtils.run_vllm_inference(vllm_request_batch)[0]
    elif backend == "openai-compat":
        raw_response = InferenceUtils.run_openai_inference(request)

    json_response = JSONUtils.extract_json_from_response(raw_response)

    return json_response["csv_table"], json_response["filename"]


def main(json_path, db_path, vector_db_path):
    """
    Example usage of the functions.
    """

    try:
        # Process JSON and store in SQL
        flatten_json_to_sql(json_path, db_path, vector_db_path)

        # Example SQL queries
        print("All artifacts:")
        print(DatabaseManager.sql_query(db_path, "SELECT * FROM document_artifacts"))

        print("\nTables only:")
        print(
            DatabaseManager.sql_query(
                db_path,
                "SELECT * FROM document_artifacts WHERE artifact_type = 'table'",
            )
        )

        # Example KNN queries
        query = "What is the average revenue per day for Meta?"
        print("\nVector index query: ", query)
        vector_query = VectorIndexManager.knn_query(query, vector_db_path)
        print(vector_query)

        # KNN + SQL
        df = DatabaseManager.sql_query(db_path, "SELECT * FROM document_artifacts")
        df = vector_query.merge(df, left_on="ids", right_on="id", how="inner")
        print("\nJoined results:")
        print(df)
    except Exception as e:
        print(f"Error in main: {e}")


if __name__ == "__main__":
    main()
