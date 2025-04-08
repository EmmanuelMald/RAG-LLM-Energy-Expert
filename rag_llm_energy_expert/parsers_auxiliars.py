import pymupdf
import pymupdf4llm
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)
from datetime import datetime
from loguru import logger
import os
import sys


sys.path.append("..")

from utils.gcp.gcs import get_file, blob_exists


def extract_pdf_content(
    pdf_path: str,
) -> dict[str, str]:
    """
    Parse a pdf that is stored in Google Cloud Storage (GCS) or in the local

    Args:
        pdf_path: str -> Either a gcs path: (ex: 'gs://bucket_name/folder_name/pdf_name.pdf') or
                        a local path (can be a relative path or a full path ex: 'local_folder/pdf_file.pdf' or 'C:Users/folder/pdf_file.pdf')

    Return:
        file_data: dict[str, str | dict] -> Dictionary with all the pdf parsed in a markdown format and metadata, it has the format:
                                            {"text": "string with all the PDF text", "metadata": <{"key": "value"}}
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
        logger.info("google cloud storage path detected")
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

        if not blob_exists(blob_name, bucket_name):
            raise ValueError(f"The file {pdf_path} does not exists")

        # Download in memory the pdf from GCS
        pdf_bytes = get_file(gcs_file_path=blob_name, bucket_name=bucket_name)

        # Load the PDF content into a Document object, each entry of the Document is a page
        pdf_document = pymupdf.Document(stream=pdf_bytes)

    # If the path seems to be a local path
    else:
        logger.info("local path detected")
        if not os.path.isfile(pdf_path):
            raise ValueError(f"The file {pdf_path} does not exists")

        # load the pdf into a Document object, each entry of the Document is a page
        pdf_document = pymupdf.Document(pdf_path)

    file_title = pdf_path.split("/")[-1].split(".")[0]

    # Reads the PDF with its metadata and creates a list of dictionaries
    logger.info("Converting PDF content to a markdown format...")
    md_text = pymupdf4llm.to_markdown(
        pdf_document,
        # page_chunks = True, # Create a list of pages of the Document
        # extract_words=True, # Adds key words to each page dictionary
        show_progress=False,
    )

    file_data = {
        "text": md_text,
        "metadata": {
            "title": file_title,
            "storage_path": pdf_path,
        },
    }

    logger.info("PDF content successfully extracted")

    return file_data


def chunk_by_md_headers(
    text: str,
) -> list[pymupdf.Document]:
    """
    Split the data by the markdown headers

    Args:
        text: str -> Text to be chunked by the markdown headers.

    Returns:
        list[pymupdf.Document] -> List of Documents, each Document has 'page_content' and 'metadata' attributes
    """
    logger.info("Chunking the text by markdown headers...")

    if not isinstance(text, str) or text == "":
        raise TypeError("The text parameter must be a not null string")

    # Markdown headers that the text will be splitted by
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
        ("####", "Header 4"),
        ("#####", "Header 5"),
    ]

    # Initialize a MarkdonHeaderTextSplitter object
    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on, strip_headers=False
    )

    # Split the text by the headers
    md_headers_chunks = markdown_splitter.split_text(text)

    logger.info("Chunks based on markdown headers successfully created")

    return md_headers_chunks


def size_md_chunks(
    md_headers_chunks: list,
    chunk_size: int,
    chunk_overlap: int,
) -> list:
    """
    Once the text has been chunked by the markdown headers, its time to split each chunk even more based on the
    embedding model specifics.

    Args:
        md_headers_chunks: list[Document] -> List of chunks (Documents) splitter by markdown headers
        chunk_size: int -> Number of tokens to split the md chunks
        chunk_overlap: int -> Number of tokens that will be overlapped on each chunk
    Returns:
        chunks_sized: list[Document] -> List of chunks (Documents) splitted by the chunk size
    """
    logger.info("Sizing markdown chunks...")

    if not isinstance(md_headers_chunks, list):
        raise TypeError("The md_headers_chunks parameter must be a list of Documents")
    if not all(isinstance(x, int) for x in [chunk_size, chunk_overlap]):
        raise TypeError("chunk_size and chunk_overlap must be integers")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )

    # Returns a list of Documents, but now each chunk has the specific size and overlap
    chunks_sized = text_splitter.split_documents(md_headers_chunks)

    logger.info("Markdown chunks sized successfully")

    return chunks_sized


def prepare_chunks_for_embeddings(
    chunks_sized: list,
    file_data: dict[str, str] | None = None,
) -> list[dict]:
    """
    Create a list of dictionaries, each dictionary contains the text to embed and its metadata

    Args:
        chunks_sized: list[Documents] -> List of Documents sized
        file_data: dict[str, str] -> Optional. Dictionary with metadata of the text
    Returns:
        list[dict] -> List of dictionaries, each dictionary is a chunk
    """
    logger.info("Preparing chunks for embedding...")

    # Error handlers for chunks_sized
    if not isinstance(chunks_sized, list):
        raise TypeError("The chunks_sized parameter must be a list of Documents")

    # Add the datetime of when the chunks were created
    extra_metadata = {
        "upload_date": datetime.now().strftime(r"%Y-%m-%d"),
    }

    # Error handlers for file_data
    if file_data is not None:
        if not isinstance(file_data, dict):
            raise TypeError("file_data parameter must be a dictionary")

        # Add all the metadata in file_data to extra_metadata
        extra_metadata.update(file_data)

    # Creating a list of dictionaries, each entry of the list is the metadata of each chunk
    final_chunks = [doc.metadata for doc in chunks_sized]

    logger.info("Creating chunks to embed...")
    for i, final_chunk in enumerate(final_chunks):
        # Add the page content in the metadata
        final_chunk["text"] = chunks_sized[i].page_content

        # Add extra_metadata to each chunk_metadata
        final_chunk.update(extra_metadata)

    logger.info("Chunks ready to be embedded")

    return final_chunks


def chunk_text(
    text: str,
    chunk_size: int,
    chunk_overlap: int,
    metadata: dict[str, str] | None = None,
) -> list[dict]:
    """
    This function is an orchestrator for the different steps that the chunking process requires

    Args:
        text: str -> String with the text to chunk
        chunk_size: int -> Number of tokens to split the md chunks
        chunk_overlap: int -> Number of tokens that will be overlapped on each chunk
        metadata: dict[str, str] -> Optional. Dictionary with the metadata of the text. Each key and value
                                              must be strings

    Return: list[dict] -> List of dictionaries, each dictionary is a chunk, the text to embed is in the key "text".
                          The other keys are metadata
    """
    # Step 1: chunk by md headers. (even when the string does not has a markdown format)
    # it already has error handlers for the 'text' parameter
    md_headers_chunks = chunk_by_md_headers(text=text)

    # Step 2: split each chunk previously created based on the chunk_size
    sized_chunks = size_md_chunks(
        md_headers_chunks=md_headers_chunks,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    # Step 3: Prepare the sized chunks to embed
    chunks_to_embed = prepare_chunks_for_embeddings(
        chunks_sized=sized_chunks, file_data=metadata
    )

    return chunks_to_embed
