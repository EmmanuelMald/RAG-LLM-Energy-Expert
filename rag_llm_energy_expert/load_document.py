from loguru import logger
import json
import sys

sys.path.append("..")

from rag_llm_energy_expert.parsers_auxiliars import (
    extract_pdf_content,
    chunk_by_md_headers,
    size_md_chunks,
    prepare_chunks_for_embeddings,
)
from rag_llm_energy_expert.config import GCP_CONFIG
from utils.gcp.gcs import upload_file_from_memory

# Initialize GCP config values
gcp_config = GCP_CONFIG()


def parse_file(
    gcs_file_path: str,
    chunk_size: int = 256,
    chunk_overlap: int = 0,
    bucket_name: str = gcp_config.BUCKET_NAME,
) -> None:
    """
    Parse a file to be embedded into vectors.

    Args:
        gcs_file_path: str -> Path to the file to be parsed. Ex. 'folder1/my_document.pdf'
        bucket_name: str -> Name of the GCS bucket
        chunk_size: int -> Number of tokens to split the text
        chunk_overlap: int -> Number of tokens that will be overlapped on each chunk

    Return:
        None
    """
    logger.info("Parsing file...")
    allowed_formats = {"pdf": extract_pdf_content}

    if not isinstance(gcs_file_path, str):
        raise ValueError(
            "gcs_file_path parameter must be a string. Ex: 'folder/my_doc.pdf'"
        )

    extension = gcs_file_path.split(".")[-1]

    if extension not in allowed_formats:
        raise ValueError(
            f"The file is in the format {extension}, which cannot be processed. Current allowed formats are: {allowed_formats.keys()}"
        )

    # Step 1: Extract the data and save it into a dictionary
    file_data = allowed_formats[extension](
        gcs_file_path=gcs_file_path, bucket_name=bucket_name
    )

    # Step 2:  Chunk the data by the markdown headers
    md_headers_chunks = chunk_by_md_headers(md_text=file_data["text"])

    # Step 3: Chunk the md_headers_chunks into smaller ones based on the chunk_size and overlap params
    chunks_sized = size_md_chunks(
        md_headers_chunks=md_headers_chunks,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    # Add the chunk_size to the metadata
    file_data["chunk_size"] = chunk_size

    # Step 4: Prepare the chunks to be embedded.
    # Returns a dictionary which each entry is a chunk
    chunks_to_embed = prepare_chunks_for_embeddings(
        chunks_sized=chunks_sized, file_data=file_data
    )

    # Step 5: Store chunks into GCP
    logger.info("Storing chunks in GCP...")
    upload_date = chunks_to_embed[0]["upload_date"]
    storage_file_path = f"chunks/{file_data['title']}_{chunk_size}_{upload_date}.txt"

    chunks_to_store = {f"chunk{i}": chunk for i, chunk in enumerate(chunks_to_embed)}

    # ensure_ascii == False to preserve word's accents
    json_chunks_plain = json.dumps(chunks_to_store, ensure_ascii=False)

    upload_file_from_memory(
        blob_name=storage_file_path,
        string_data=json_chunks_plain,
        bucket_name=bucket_name,
    )

    logger.info(f"Chunks stored in {storage_file_path}")

    return
