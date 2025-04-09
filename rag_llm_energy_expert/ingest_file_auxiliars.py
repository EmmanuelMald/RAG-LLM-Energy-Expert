import pymupdf
from langchain_text_splitters import TokenTextSplitter
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer
import numpy as np
import uuid
from datetime import datetime
from loguru import logger
from typing import Union
import os

import sys

sys.path.append("..")

from utils.gcp.gcs import get_file


def extract_pdf_content(
    pdf_path: str,
) -> dict[str, Union[str | dict]]:
    """
    Parse a pdf that is stored in Google Cloud Storage (GCS) or in the local

    Args:
        pdf_path: str -> Either a gcs path: (ex: 'gs://bucket_name/folder_name/pdf_name.pdf') or
                        a local path (can be a relative path or a full path ex: 'local_folder/pdf_file.pdf' or 'C:Users/folder/pdf_file.pdf')

    Return:
        file_data: Union[str | dict] -> Dictionary with all the pdf parsed in a markdown format and metadata, it has the format:
                                            {"text": "string with all the PDF text", "metadata": {"key": "value"}}
    """
    logger.info("Extracting PDF content...")
    # Datatype Check
    if not isinstance(pdf_path, str):
        raise ValueError(
            "pdf_path parameter must be a string containing either a gcs path (ex: 'gs://bucket_name/folder_name/pdf_name.pdf')"
            " or a local path (can be a relative path or a full path ex: 'local_folder/pdf_file.pdf' or 'C:Users/folder/pdf_file.pdf')"
        )

    # File extension check
    file_extension = pdf_path.split(".")[-1]

    if file_extension != "pdf":
        raise TypeError("The file is not a pdf")

    # Checking if the path comes from gcs or is a local path
    if pdf_path.startswith("gs://"):
        logger.info("Google cloud storage path detected")
        # Taking out the 'gs://' part of the string
        useful_pdf_path = pdf_path[5:]

        # Splitting the pdf path in two
        pdf_path_parts = useful_pdf_path.split("/", maxsplit=1)

        try:
            bucket_name = pdf_path_parts[0]
            blob_name = pdf_path_parts[1]

        except Exception as e:
            raise ValueError(
                "There is an error in the pdf_path parameter. if you try to set a gcs file path,"
                f" use the following format: 'gs://bucket_name/path/to/file.pdf'. {e}"
            )

        # Download in memory the pdf from GCS
        pdf_bytes = get_file(gcs_file_path=blob_name, bucket_name=bucket_name)

        # Load the PDF content into a Document object, each entry of the Document is a page
        pdf_document = pymupdf.Document(stream=pdf_bytes)

    # If the path seems to be a local path
    else:
        logger.info("Local path detected")
        if not os.path.isfile(pdf_path):
            raise ValueError(f"The file {pdf_path} does not exists")

        # load the pdf into a Document object, each entry of the Document is a page
        pdf_document = pymupdf.Document(pdf_path)

    file_title = pdf_path.split("/")[-1].split(".")[0]

    pdf_text = "\n".join(
        [
            pdf_document.get_page_text(pno=page_num)
            for page_num in range(len(pdf_document))
        ]
    )

    file_data = {
        "text": pdf_text,
        "metadata": {
            "title": file_title,
            "storage_path": pdf_path,
            "upload_date": datetime.now().strftime(r"%Y-%m-%d"),
        },
    }

    logger.info("PDF content successfully extracted")

    return file_data


def chunk_text(
    text: str,
    embedding_model: str,
    chunk_overlap: int,
) -> list[np.ndarray]:
    """
    Split the data into chunks based on the embedding model used

    Args:
        text: str -> Text to be chunked by the markdown headers.
        embedding_model: str -> Name of the embedding model to use. Must be available in sentence transformers
        chunk_overlap: int -> Number of tokens to overlap between chunks.

    Returns:
        list[np.ndarray] -> List of embeddings, each entry is a vector representing the string of each chunk
    """
    logger.info("Chunking text...")

    parameters = [text, embedding_model]

    if not all([(isinstance(x, str)) & (x != "") for x in parameters]):
        raise TypeError(
            "The parameters 'embedding_model' and 'text' must be not null strings"
        )

    if (not isinstance(chunk_overlap, int)) | (chunk_overlap < 0):
        raise ValueError("Chunk_overlap must be an integer greater or equal to 0")

    try:
        model = SentenceTransformer(embedding_model, trust_remote_code=True)

    except Exception as e:
        raise ValueError(
            f"Error loading the embedding model from sentence transformers: {e}"
        )

    # Generation of a tokenizer based on the embedding model selected
    tokenizer = AutoTokenizer.from_pretrained(embedding_model)

    # Creation of a TokenTextSplitter object
    splitter = TokenTextSplitter.from_huggingface_tokenizer(
        tokenizer,
        chunk_size=model.max_seq_length,  # chunk_size in this case is measured in tokens
        chunk_overlap=chunk_overlap,
    )

    # Generate a list of strings, each string is a chunk
    chunks = splitter.split_text(text)

    logger.info("text successfully chunked")

    return chunks


def embed_chunks(
    chunks: list[str],
    embedding_model: str,
    metadata: Union[dict[str, str], None] = None,
) -> list[dict]:
    """
    Embed string chunks into vectors based on the embedding model used

    Args:
        chunks: list[str] -> List of strings. Each string must comply with the embedding model
                            chunk size limit
        metadata: Union[dict[str, str], None] -> Dictionary of metadata to be inserted to each chunk
        embedding_model: str -> Name of the embedding model to use. Must be available in sentence transformers

    Return:
        list[dict] -> List of dictionaries, each dictionary is a chunk, the structure of the dictionary is:
                        'vector': np.ndarray,
                        'vector_id': uuid string,
                        'payload': dictionary with two values:
                                    'text' -> String embedded in a vector
                                    'metadata' -> Dictionary with text's metadata
    """
    logger.info("Embedding chunks...")

    if not isinstance(chunks, list):
        raise TypeError("'chunks' parameter must be a list of np.ndarray Objects")
    else:
        if not all([(isinstance(x, str)) & (x != "") for x in chunks]):
            raise TypeError("Each entry of the chunk list must be a not null string")

    if not isinstance(metadata, Union[dict, None]):
        raise TypeError("'metadata' must be a dictionary of strings")

    if (not isinstance(embedding_model, str)) | (embedding_model == ""):
        raise TypeError(
            "'embedding_model' must be the name of an embedding model available on sentence-transformers"
        )

    try:
        model = SentenceTransformer(embedding_model, trust_remote_code=True)

    except Exception as e:
        raise ValueError(
            f"Error loading the embedding model from sentence transformers: {e}"
        )

    # Embedding the chunk text using batch embedding
    chunks_embedded = model.encode(chunks)

    # Create a list of dictionaries, which each dictionary is a chunk with all the necessary to be
    # indexed into a vector DB
    final_chunks = [
        {
            "id": str(uuid.uuid4()),
            "vector": chunks_embedded[i],
            "payload": {
                "text": chunk_text,
                "metadata": metadata,
            },
        }
        for i, chunk_text in enumerate(chunks)
    ]

    logger.info("Chunks successfully embedded")

    return final_chunks
