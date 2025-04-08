from sentence_transformers import SentenceTransformer
from qdrant_client.models import PointStruct
from loguru import logger
import uuid
import sys

sys.path.append("..")

from rag_llm_energy_expert.parsers_auxiliars import (
    extract_pdf_content,
    chunk_text,
)
from rag_llm_energy_expert.config import GCP_CONFIG, QDRANT_CONFIG
from utils.vector_db.qdrant import update_points

# Initialize config classes
gcp_config = GCP_CONFIG()
qdrant_config = QDRANT_CONFIG()


def parse_file(
    file_path: str,
    chunk_size: int,
    chunk_overlap: int,
) -> list[dict[str, str]]:
    """
    Parse a file to be embedded into vectors.

    Args:
        file_path: str -> Either a gcs path: (ex: 'gs://bucket_name/folder_name/pdf_name.pdf') or
                        a local path (can be a relative path or a full path ex: 'local_folder/pdf_file.pdf' or 'C:Users/folder/pdf_file.pdf')
        chunk_size: int -> Number of tokens to split the text
        chunk_overlap: int -> Number of tokens that will be overlapped on each chunk

    Return:
        list[dict[str, str]] -> List of dictionaries, each dictionary is a chunk, which contains only two keys: 'text' contains the text to embedded
                                and 'metadata' which is also a dictionary
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

    # Step 2:  Chunk the data
    chunks = chunk_text(
        text=file_data["text"],
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        metadata=file_data["metadata"],
    )

    return chunks


def create_points(
    chunks: list[dict], embedding_model: SentenceTransformer
) -> list[PointStruct]:
    """
    From the chunks created (list of dictionaries), create vectors based on the embedding model selected, and generate
    a list of PointStruct ready to index into the Qdrant vector database

    Args:
        chunks: list[dict] -> list of Dictionaries, where each dictionary is a chunk. Each dictionary contains the key 'text'
                              which is the text to be embedded, and the other keys are metadata
        embedding_model: SentenceTransformer -> SentenceTransformer instance

    Returns:
        list[PointStruct] -> Returns a list of PointStruct, which is ready to be indexed into the vector database
    """
    logger.info("Creating points from chunks file...")
    if not isinstance(embedding_model, SentenceTransformer):
        raise TypeError(
            "The parameter embedding_model must be a SentenceTransformer instance"
        )

    if not isinstance(chunks, list):
        raise TypeError("The 'chunks' parameter must be a list of dictionaries")

    elif not all([isinstance(x, dict) for x in chunks]):
        raise TypeError("All the entries of the chunks list must be dictionaries")

    logger.info("Embedding chunks...")
    points = list()

    # Create a list where each entry is the text to encode for each chunk
    chunks_text = [chunk_info["text"] for chunk_info in chunks]

    # SentenceTransformers allows batch embeddings
    chunks_vectors = embedding_model.encode(chunks_text)

    # Create a list of PointStruct objects, each PointStruct object is a chunk
    points = [
        PointStruct(
            id=str(uuid.uuid4()),
            vector=chunks_vectors[chunk_number],
            payload=chunk_info,
        )
        for chunk_number, chunk_info in enumerate(chunks)
    ]

    logger.info("Embeddings created")

    return points


def upload_document(
    file_path: str,
    chunk_overlap: int,
    vectordb_collection: str,
    embedding_model: str,
):
    """
    Reads a file stored in GCS or in the local, then parse it, chunk it, generate vectors for each chunk, and then embed those vectors
    into a vector database collection.

    Args:
        file_path: str -> Either a gcs path: (ex: 'gs://bucket_name/folder_name/pdf_name.pdf') or
                        a local path (can be a relative path or a full path ex: 'local_folder/pdf_file.pdf' or 'C:Users/folder/pdf_file.pdf')
        vectordb_collection: str -> Name of the vector DB collection where the document's chunks will be stored. Default qdrant_config.COLLECTION_NAME + qdrant_config.COLLECTION_VERSION
        chunk_overlap: int -> Number of tokens to overlap between chunks. Default 0
        embedding_model: str -> Name of the embedding model to be used. Must be available in the sentence-transformers library.
                                Default: "sentence-transformers/all-MiniLM-L6-v2"

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
    chunks = parse_file(
        file_path=file_path,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    # Step 2: Create the vector embeddings
    points = create_points(
        chunks=chunks,
        embedding_model=model,
    )

    # Step 3: Store the vectors into the vector db
    update_points(
        collection_name=vectordb_collection,
        points=points,
    )
