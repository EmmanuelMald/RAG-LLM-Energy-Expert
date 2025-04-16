from loguru import logger
import requests
import sys

sys.path.append("../../..")

from rag_llm_energy_expert.config import GCPConfig
from rag_llm_energy_expert.credentials import generate_id_token
from rag_llm_energy_expert.services.ingestion.parsers.pdf_parser import parse_pdf_file
from rag_llm_energy_expert.utils.vector_db.qdrant import (
    create_points,
    create_collection,
    update_points,
)

gcp_config = GCPConfig()


def main(
    file_path: str,
    collection_name: str,
    embedding_model_name: str = None,
    chunk_overlap: int = None,
    create_db_collection: bool = False,
) -> None:
    """
    Ingest a PDF into a vector DB

    Args:
        file_path: str -> Either a gcs path: (ex: 'gs://bucket_name/folder_name/pdf_name.pdf') or
                        a local path (can be a relative path or a full path ex: 'local_folder/pdf_file.pdf' or 'C:Users/folder/pdf_file.pdf')
        embedding_model: str -> Name of the embedding model to use. Must be available in sentence transformers
        chunk_overlap: int -> Number of tokens that will be overlapped on each chunk
        collection_name: str -> Name of the vector db collection where the chunks will be indexed
        create_collection: bool -> If the collection does not exists, creates it if create_collection == True

    Return:
        None
    """
    logger.info("Parsing file...")
    allowed_formats = {"pdf": parse_pdf_file}

    if not isinstance(file_path, str) or file_path == "":
        raise ValueError("file_path must be a not null string")

    extension = file_path.split(".")[-1]

    if extension not in allowed_formats:
        raise ValueError(
            f"The file is in the format {extension}, which cannot be processed. Current allowed formats are: {', '.join(allowed_formats.keys())}"
        )

    # Step 1: Extract the data and save it into a dictionary
    file_data = allowed_formats[extension](file_path)

    # Step 2: Generate embeddings from the PDF text
    logger.info("Generating embeddings...")

    # Generate the ID Token to use the embedding service
    embedding_service_token = generate_id_token(
        audience=gcp_config.EMBEDDING_SERVICE_URL
    )

    headers = {"Authorization": f"Bearer {embedding_service_token}"}

    payload = {
        "text": file_data["text"],
        "metadata": file_data["metadata"],
        "chunk_overlap": chunk_overlap,
        "embedding_model_name": embedding_model_name,
    }

    embed_text_url = gcp_config.EMBEDDING_SERVICE_URL + gcp_config.EMBED_TEXT_ENDPOINT

    try:
        embeddings_response = requests.post(
            url=embed_text_url, json=payload, headers=headers
        )
    except Exception as e:
        raise ValueError(f"There was an error during the embeddings generation: {e}")

    if embeddings_response.status_code != 200:
        raise ValueError(
            "There was an error during the embeddings generation. "
            f"Status code: {embeddings_response.status_code}. "
            f"Response: {embeddings_response.text}"
        )
    else:
        # The embed-text endpoint returns a dictionary with the key chunks, which value is a list
        # of dictionaries
        logger.info("Embeddings generated")
        chunks = embeddings_response.json()["chunks"]
        vector_dimension = len(chunks[0]["vector"])

    # Step 3: Prepare chunks to be indexed in the Qdrant vector DB
    qdrant_points = create_points(chunks=chunks)

    # Step 4: Create the vector DB collection if needed
    if create_db_collection:
        create_collection(collection_name=collection_name, vector_size=vector_dimension)

    # Step 5: Upload the qdrant points into the qdrant collection
    update_points(collection_name=collection_name, points=qdrant_points)


if __name__ == "__main__":
    main()
