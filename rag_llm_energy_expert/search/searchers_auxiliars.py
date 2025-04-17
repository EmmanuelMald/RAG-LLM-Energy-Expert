from loguru import logger
from qdrant_client import models
from typing import Union
import requests

import sys

sys.path.append("../../../..")

from rag_llm_energy_expert.credentials import get_gcp_config


gcp_config = get_gcp_config()


def process_query(
    query: str,
    documents_limit: int,
    embedding_model_name: Union[str, None],
    chunk_overlap: Union[int, None],
) -> list[models.QueryRequest]:
    """
    Process the user's query before making a search on the vector DB.
    In case the query is too long, it splits the text and for each chunk generated, return
    a models.QueryRequest object.

    Args:
        query: str -> User's query
        documents_limit: Union[int, None] -> Limit of documents retrieved by the search
        embedding_model_name: Union[str, None] -> Name of the embedding model to generate the embeddings. Must match with the
                                     embedding model that the documents were embedded in the vector DB
        chunk_overlap: Union[int, None] -> Number of tokens to overlap the chunks

    Return:
        search_queries: list[models.QueryRequest] -> List of QueryRequests ready for vector search
    """
    logger.info("Preprocessing query...")

    # Error handlers for query
    if not isinstance(query, str) or query == "":
        raise ValueError("The parameter 'query' must be a non empty string")

    # Error handlers for embedding_model_name
    if not isinstance(embedding_model_name, Union[str, None]):
        raise ValueError("embedding_model_name must be a not empty string or None")
    elif embedding_model_name == "":
        raise ValueError("'embedding_model_name' must not be an empty string")

    # Error hander for documents_limit
    if not isinstance(documents_limit, int) or documents_limit < 1:
        raise ValueError("'documents_limit' must be an integer greater or equal than 1")

    if not isinstance(chunk_overlap, Union[int, None]):
        raise ValueError("'chunk_overlap' must be an integer greater or equal than 1")
    elif isinstance(chunk_overlap, int) and chunk_overlap < 0:
        raise ValueError("'chunk_overlap' must be greater or equal than 1")

    # Use the embedding service deployed on CloudRun to generate the embeddings
    payload = {
        "text": query,
        "embedding_model_name": embedding_model_name,
        "chunk_overlap": chunk_overlap,
    }

    # Generating the token to authenticate the request to the embedding service
    token = gcp_config.EMBEDDING_SERVICE_IDTOKEN.get_secret_value()
    headers = {"Authorization": f"Bearer {token}"}

    full_text_embedding_url = (
        gcp_config.EMBEDDING_SERVICE_URL + gcp_config.EMBED_TEXT_ENDPOINT
    )

    try:
        logger.info("Generating embeddings...")
        response = requests.post(
            url=full_text_embedding_url, json=payload, headers=headers
        )
    except Exception as e:
        raise ValueError(f"There was an error using the embedding service: {e}")

    if response.status_code != 200:
        raise ValueError(
            f"Bad request to the embedding service: Status code: {response.status_code}. "
            f"{response.text}"
        )

    logger.info("Embeddings generated successfully")
    # Get the list of chunks generated
    embeddings = response.json()["chunks"]

    # For each chunk generated, get its vector
    vectors = [chunk["vector"] for chunk in embeddings]

    # Prepare the vectors obtained to be used in the vector DB
    logger.info("Preparing embeddings for vector search")
    search_queries = [
        models.QueryRequest(
            query=vector,
            with_payload=True,
            with_vector=False,
            limit=documents_limit,
        )
        for vector in vectors
    ]

    logger.info("Query preprocessed successfully")
    return search_queries


def process_query_results(results: list[models.models.QueryResponse]) -> str:
    """
    Return the query responses for each QueryRequest generated

    Args:
        results: list[models.models.QueryResponse] -> List of QueryResponses obtained after the semantic search

    Returns:
        str -> String with all the text of the documents retrieved
    """
    logger.info("Processing query results...")
    full_text = ""

    for query_response in results:
        # Get a list of points for each QueryResponse
        query_response_points = query_response.points

        # Extract the text of each point
        for point in query_response_points:
            full_text += point.payload["text"] + "\n\n"

    logger.info("Query results processed")
    return full_text
