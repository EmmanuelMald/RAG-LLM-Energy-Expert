from langchain_text_splitters import MarkdownHeaderTextSplitter, MarkdownTextSplitter
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer
import uuid
from loguru import logger
from typing import Union


def chunk_text(
    text: str,
    embedding_model: SentenceTransformer,
    embedding_model_name: str,
    chunk_overlap: int,
) -> list[str]:
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
        list[str] -> List of strings, each entry is chunked text that can be embedded by the embedding model
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
    logger.info("Generating the tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(embedding_model_name)

    # Initialize the MarkdownHeaderTextSplitter instance.
    logger.info("Initializing the MarkdownHeaderTextSplitter instance...")
    headers_to_split_on = [
        ("##", 2),
        ("###", 3),
        ("####", 4),
        ("#####", 5),
        ("######", 6),
    ]
    markdown_header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on, strip_headers=False
    )

    # initialize_the MarkdownTextSplitter instance.
    logger.info("Initializing the MarkdownTextSplitter instance...")
    markdown_text_splitter = MarkdownTextSplitter.from_huggingface_tokenizer(
        tokenizer=tokenizer,
        chunk_size=embedding_model.max_seq_length,
        chunk_overlap=chunk_overlap,
    )

    logger.info("Splitting text with the MarkdownHeaderTextSplitter instance...")
    # Splits the text based on the headers. This split generates a list of Documents
    docs_markdownheaders = markdown_header_splitter.split_text(text)

    # Getting the splitted text from the Document objects
    chunks_markdownheaders = [doc.page_content for doc in docs_markdownheaders]

    logger.info("Splitting text with the MarkdownTextSplitter instance...")
    # Instance a list to store the final
    chunks = list()

    for chunk in chunks_markdownheaders:
        # Splits the text based on the max tokens supported by the embedding model
        chunks.extend(markdown_text_splitter.split_text(chunk))

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
            "vector": chunks_embedded[i].tolist(),
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
    logger.info(f"Loading the embedding model: {embedding_model_name}")
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
