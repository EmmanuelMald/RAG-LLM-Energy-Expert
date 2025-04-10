from langchain_text_splitters import TokenTextSplitter
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer
import numpy as np
import uuid
from loguru import logger
from typing import Union


def chunk_text(
    text: str,
    embedding_model: SentenceTransformer,
    embedding_model_name: str,
    chunk_overlap: int,
) -> list[np.ndarray]:
    """
    Split the data into chunks based on the embedding model used.
    The embedding_model and embedding_model_name are both required, embedding_model to only call the SentenceTransformer instance once,
    and the embedding_model_name because the model card of the instances might not be filled to extract the name from there.

    Args:
        text: str -> Text to be chunked by the markdown headers.
        embedding_model: SentenceTransformer -> SentenceTransformer instance that will be used as part of the Tokenizer instance
        embedding_model_name: str -> Name of the embedding model. This is necessary due to some models does not has its model card filled
        chunk_overlap: int -> Number of tokens to overlap between chunks.

    Returns:
        list[np.ndarray] -> List of embeddings, each entry is a vector representing the string of each chunk
    """
    logger.info("Chunking text...")

    parameters = [text, embedding_model_name]

    if not all([(isinstance(x, str)) & (x != "") for x in parameters]):
        raise TypeError(
            "The parameter 'text' and 'embedding_model_name' must be not null strings"
        )

    if not isinstance(embedding_model, SentenceTransformer):
        raise TypeError(
            "The parameter 'embedding_model' must be a SentenceTransformer instance"
        )

    if (not isinstance(chunk_overlap, int)) | (chunk_overlap < 0):
        raise ValueError(
            "The parameter 'chunk_overlap' must be an integer greater or equal to 0"
        )

    # Generation of a tokenizer based on the embedding model selected
    tokenizer = AutoTokenizer.from_pretrained(embedding_model_name)

    # Creation of a TokenTextSplitter object
    splitter = TokenTextSplitter.from_huggingface_tokenizer(
        tokenizer,
        chunk_size=embedding_model.max_seq_length,  # chunk_size in this case is measured in tokens
        chunk_overlap=chunk_overlap,
    )

    # Generate a list of strings, each string is a chunk
    chunks = splitter.split_text(text)

    logger.info("text successfully chunked")

    return chunks


def embed_chunks(
    chunks: list[str],
    embedding_model: SentenceTransformer,
    metadata: Union[dict[str, str], None] = None,
) -> list[dict]:
    """
    Embed string chunks into vectors based on the embedding model used

    Args:
        chunks: list[str] -> List of strings. Each string must comply with the embedding model
                            chunk size limit
        metadata: Union[dict[str, str], None] -> Dictionary of metadata to be inserted to each chunk
        embedding_model: SentenceTransformer -> SentenceTransformer instance that will be used to embed the text

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

    if not isinstance(embedding_model, SentenceTransformer):
        raise TypeError("'embedding_model' must be a SentenceTransformer instance")

    # Embedding the chunk text using batch embedding
    chunks_embedded = embedding_model.encode(chunks)

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


def text_embedder(
    text: str,
    chunk_overlap: str,
    embedding_model_name: str,
    metadata: Union[dict[str, str], None] = None,
) -> list[dict]:
    """
    Function that combines the chunking and embedding of text,

    Args:
        text: str -> Text to be chunked by the markdown headers.
        chunk_overlap: int -> Number of tokens to overlap between chunks.
        embedding_model_name: str -> Name of the embedding model to use. Must be available in sentence transformers
        metadata: Union[dict[str, str], None] -> Dictionary of metadata to be inserted to each chunk

    Return:
        list[dict] -> List of dictionaries, each dictionary is a chunk, the structure of the dictionary is:
                        'vector': np.ndarray,
                        'vector_id': uuid string,
                        'payload': dictionary with two values:
                                    'text' -> String embedded in a vector
                                    'metadata' -> Dictionary with text's metadata
    """
    # Error handlers for the embedding_model_name only, the other parameters are
    # tested in other functions
    if not isinstance(embedding_model_name, str) or embedding_model_name == "":
        raise TypeError(
            "The parameter 'embedding_model_name' must be a not null string"
        )

    # Initialize the model
    try:
        model = SentenceTransformer(embedding_model_name, trust_remote_code=True)
    except Exception as e:
        raise ValueError(
            f"Error loading the embedding model from sentence transformers: {e}"
        )

    # Chunking the text based on the max tokens supported by the model
    text_chunked = chunk_text(
        text=text,
        embedding_model=model,
        embedding_model_name=embedding_model_name,
        chunk_overlap=chunk_overlap,
    )

    # Embedding the text based on the embedding model
    text_embedded = embed_chunks(
        chunks=text_chunked,
        embedding_model=model,
        metadata=metadata,
    )

    return text_embedded
