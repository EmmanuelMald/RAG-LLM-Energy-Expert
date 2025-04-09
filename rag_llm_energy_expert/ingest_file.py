from loguru import logger

import sys

sys.path.append("..")

from rag_llm_energy_expert.ingest_file_auxiliars import (
    extract_pdf_content,
    chunk_text,
    embed_chunks,
)
from utils.vector_db.qdrant import update_points, create_points


def upload_file(
    file_path: str,
    embedding_model: str,
    chunk_overlap: int,
    collection_name: str,
) -> None:
    """
    Parse a file to be embedded into vectors.

    Args:
        file_path: str -> Either a gcs path: (ex: 'gs://bucket_name/folder_name/pdf_name.pdf') or
                        a local path (can be a relative path or a full path ex: 'local_folder/pdf_file.pdf' or 'C:Users/folder/pdf_file.pdf')
        embedding_model: str -> Name of the embedding model to use. Must be available in sentence transformers
        chunk_overlap: int -> Number of tokens that will be overlapped on each chunk
        collection_name: str -> Name of the vector db collection where the chunks will be indexed

    Return:
        None
    """
    logger.info("Parsing file...")
    allowed_formats = {"pdf": extract_pdf_content}

    if not isinstance(file_path, str) or file_path == "":
        raise ValueError("file_path must be a not null string")

    extension = file_path.split(".")[-1]

    if extension not in allowed_formats:
        raise ValueError(
            f"The file is in the format {extension}, which cannot be processed. Current allowed formats are: {', '.join(allowed_formats.keys())}"
        )

    # Step 1: Extract the data and save it into a dictionary
    file_data = allowed_formats[extension](file_path)

    # Step 2:  Chunk the data extracted
    # This function already has error handlers for its parameters
    chunks = chunk_text(
        text=file_data["text"],
        embedding_model=embedding_model,
        chunk_overlap=chunk_overlap,
    )

    # Step 3: Embed chunks
    embedded_chunks = embed_chunks(
        chunks=chunks,
        embedding_model=embedding_model,
        metadata=file_data["metadata"],
    )

    # Step 4: Prepare chunks to be indexed in the Qdrant vector DB
    qdrant_points = create_points(chunks=embedded_chunks)

    # Step 5: Upload the qdrant points into the qdrant collection
    update_points(collection_name=collection_name, points=qdrant_points)
