from sentence_transformers import SentenceTransformer
from qdrant_client.models import PointStruct
from loguru import logger
import uuid
import json
import sys

sys.path.append("..")

from rag_llm_energy_expert.parsers_auxiliars import (
    extract_pdf_content,
    chunk_by_md_headers,
    size_md_chunks,
    prepare_chunks_for_embeddings,
)
from rag_llm_energy_expert.config import GCP_CONFIG, QDRANT_CONFIG
from utils.gcp.gcs import upload_file_from_memory, get_file
from utils.vector_db.qdrant import update_points

# Initialize config classes
gcp_config = GCP_CONFIG()
qdrant_config = QDRANT_CONFIG()


def parse_file(
    bucket_name: str,
    gcs_document_path: str,
    chunk_size: int,
    chunk_overlap: int,
) -> str:
    """
    Parse a file to be embedded into vectors.

    Args:
        gcs_document_path: str -> GCS Path to the file to be parsed. Ex. 'folder1/my_document.pdf'
        bucket_name: str -> Name of the GCS bucket
        chunk_size: int -> Number of tokens to split the text
        chunk_overlap: int -> Number of tokens that will be overlapped on each chunk

    Return:
        str -> GCS path where the chunks were stored. Ex: "folder1/folder2/chunks_data.txt"
    """
    logger.info("Parsing file...")
    allowed_formats = {"pdf": extract_pdf_content}

    if not isinstance(gcs_document_path, str):
        raise ValueError(
            "gcs_document_path parameter must be a string. Ex: 'folder/my_doc.pdf'"
        )

    extension = gcs_document_path.split(".")[-1]

    if extension not in allowed_formats:
        raise ValueError(
            f"The file is in the format {extension}, which cannot be processed. Current allowed formats are: {allowed_formats.keys()}"
        )

    # Step 1: Extract the data and save it into a dictionary
    file_data = allowed_formats[extension](
        gcs_file_path=gcs_document_path, bucket_name=bucket_name
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

    return storage_file_path


def create_points(
    bucket_name: str, chunks_file: str, embedding_model: SentenceTransformer
) -> list[PointStruct]:
    """
    Load a txt file containing the chunk information, create vectors based on the embedding model selected, and generate
    a list of PointStruct ready to index into the Qdrant vector database

    Args:
        chunks_file: str -> String with the GCS path where the txt file is stored. Ex: "gcs_folder1/gcs_folder2/chunks.txt"
                           This file must contain a dictionary of chunks, each chunk is also a dictionary with the text
                           to be embedded and its metadata
                                    Ex of the txt file structure:
                                    {
                                    "chunk0": {"text":"text_to_embed", "chunk_size": 250, "title": "name_of_the_document"},
                                    "chunk1": {"text": "text_to_embed2", "chunk_size": 250, "title", "name_of_the_document"},
                                    }
        embedding_model: SentenceTransformer -> SentenceTransformer instance

    Returns:
        list[PointStruct] -> Returns a list of PointStruct, which is ready to be indexed into the vector database
    """
    logger.info("Creating points from chunks file...")
    if not isinstance(embedding_model, SentenceTransformer):
        raise TypeError(
            "The parameter embedding_model must be a SentenceTransformer instance"
        )

    logger.info("Downloading chunks...")

    # get_file already has error handlers for its parameters
    # chunks is in the format: {"chunk0": {"text": "text to encode", "title": "title of the document", ...}, ...}
    chunks = json.loads(get_file(chunks_file, bucket_name))
    logger.info("Chunks downloaded successfully")

    logger.info("Embedding chunks...")
    points = list()

    # Create a list where each entry is the text to encode for each chunk
    chunks_text = [chunk_info["text"] for chunk_info in chunks.values()]

    # SentenceTransformers allows batch embeddings
    chunks_vectors = embedding_model.encode(chunks_text)

    # Create a list of PointStruct objects, each PointStruct object is a chunk
    points = [
        PointStruct(
            id=str(uuid.uuid4()),
            vector=chunks_vectors[chunk_number],
            payload=chunk_info,
        )
        for chunk_number, chunk_info in enumerate(chunks.values())
    ]

    logger.info("Embeddings created")

    return points


def upload_document(
    gcs_document_path: str,
    chunk_overlap: int,
    vectordb_collection: str,
    embedding_model: str,
    bucket_name: str,
):
    """
    Reads a file stored in GCS, then parse it, chunk it, generate vectors for each chunk, and then embed those vectors
    into a vector database collection.

    Args:
        gcs_document_path: str -> GCS path where the document is stored. Ex: "folder1/folder2/file_name.pdf"
        vectordb_collection: str -> Name of the vector DB collection where the document's chunks will be stored. Default qdrant_config.COLLECTION_NAME + qdrant_config.COLLECTION_VERSION
        chunk_overlap: int -> Number of tokens to overlap between chunks. Default 0
        embedding_model: str -> Name of the embedding model to be used. Must be available in the sentence-transformers library.
                                Default: "sentence-transformers/all-MiniLM-L6-v2"
        bucket_name: str -> Name of the GCS bucket where the document is stored. Default: gcp_config.BUCKET_NAME

    Return:
        None
    """
    # Error handler from embedding_model, the other parameters will be tested in further functions
    if not isinstance(embedding_model, str) or embedding_model == "":
        raise ValueError("The embedding_model parameter must be a not empty string")

    try:
        model = SentenceTransformer(embedding_model)
        logger.info("embedding model successfully initialized")

    except Exception as e:
        raise ValueError(
            f"Error loading the embedding model from sentence transformers: {e}"
        )

    # Get the max tokens supported by the model
    chunk_size = model.max_seq_length

    # Step 1: Parse file
    chunks_file_path = parse_file(
        bucket_name=bucket_name,
        gcs_document_path=gcs_document_path,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    # Step 2: Create the vector embeddings
    points = create_points(
        bucket_name=bucket_name,
        chunks_file=chunks_file_path,
        embedding_model=model,
    )

    # Step 3: Store the vectors into the vector db
    update_points(
        collection_name=vectordb_collection,
        points=points,
    )
