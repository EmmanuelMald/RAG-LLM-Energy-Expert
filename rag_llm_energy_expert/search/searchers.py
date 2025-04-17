from qdrant_client import QdrantClient

import sys

sys.path.append("../../../..")

from rag_llm_energy_expert.search.searchers_auxiliars import (
    process_query,
    process_query_results,
)
from rag_llm_energy_expert.credentials import get_qdrant_config

qdrant_config = get_qdrant_config()

qdrant_client = QdrantClient(
    url=qdrant_config.URL, api_key=qdrant_config.API_KEY.get_secret_value()
)


def semantic_search(
    query: str,
    embedding_model_name: str,
    chunk_overlap: int,
    collection_name: str,
    documents_limit: int,
):
    """
    Generate the necessary steps to do the semantic search of the user's query, and retrieve the
    documents with the most relevant information.

    Args:
        query: str -> User's query
        documents_limit: Union[int, None] -> Limit of documents retrieved by the search
        embedding_model_name: Union[str, None] -> Name of the embedding model to generate the embeddings. Must match with the
                                     embedding model that the documents were embedded in the vector DB
        chunk_overlap: Union[int, None] -> Number of tokens to overlap the chunks
        collection_name: str -> Name of the vector DB collection where the documents will be retrieved

    Return:
        str -> All the document's text
    """
    # Get a list of vector queries
    # Already has error handlers
    search_queries = process_query(
        query=query,
        embedding_model_name=embedding_model_name,
        chunk_overlap=chunk_overlap,
        documents_limit=documents_limit,
    )

    # Do semantic search
    results = qdrant_client.query_batch_points(
        collection_name=collection_name,
        requests=search_queries,
    )

    # Get a list of results from the query batch
    # Already has error handlers
    data_retrieved = process_query_results(results=results)

    return data_retrieved
