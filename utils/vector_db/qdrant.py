from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
    FilterSelector,
)
from loguru import logger
import sys

sys.path.append("../..")

from rag_llm_energy_expert.credentials import get_qdrant_config

# Initialize a QDRANT_CONFIG class
config = get_qdrant_config()

# Initialize a general Qdrant client
client = QdrantClient(url=config.URL, api_key=config.API_KEY.get_secret_value())


def document_in_collection(collection_name: str, document_title: str) -> bool:
    """
    Returns True if the document is already in the vector DB collection

        Args:
            collection_name: str -> Name of the vector DB collection
            document_title: str -> Title of the document to search

        Return:
            bool: True if document is in collection, False if not
    """
    parameters = [collection_name, document_title]

    # Error handlers for parameters
    if not all([(isinstance(x, str)) & (x != "") for x in parameters]):
        raise TypeError(
            "The parameters collection_name and document_title must be not null strings"
        )

    if not client.collection_exists(collection_name):
        raise ValueError(
            f"The collection {collection_name} does not exists. To create it, please"
            " use the create_collection function"
        )

    # Create a filter to match payload
    title_filter = Filter(
        must=[FieldCondition(key="title", match=MatchValue(value=document_title))]
    )

    # Scroll through all matching vectors
    scroll_result = client.scroll(
        collection_name=collection_name,
        scroll_filter=title_filter,
        limit=1,  # In this case, I only need 1 vector to know if the document is already indexed
    )

    # Access the points
    vectors = scroll_result[0]  # List of PointStruct

    if len(vectors) > 0:
        return True

    else:
        return False


def delete_document(collection_name: str, document_title: str) -> None:
    """
    Deletes a document from the collection if exists

    Args:
        collection_name: str -> Name of the vector db's collection
        document_title: str -> Title of the document to delete

    Return:
        None
    """
    logger.info("Deleting document...")
    # document_in_collection already has error handlers for its parameters
    if not document_in_collection(collection_name, document_title):
        raise ValueError(
            f"The document {document_title} is not in the vector db collection {collection_name}"
        )

    # Delete document
    client.delete(
        collection_name=collection_name,
        points_selector=FilterSelector(
            filter=Filter(
                must=FieldCondition(
                    key="title",
                    match=MatchValue(value=document_title),
                )
            ),
        ),
    )

    logger.info(f"Document {document_title} deleted")


def upload_points(collection_name: str, points: list[PointStruct]) -> None:
    """
    Upload the points generated when parsing and chunking a document to the vector db collection

    Args:
        collection_name: str -> Name of the collection
        points: list[PointStruct] -> List of PointStruct objects, each PointStruct is a chunk of a document

    Return: None
    """

    logger.info("Uploading points...")
    # Error handlers for parameters
    if not isinstance(collection_name, str) or collection_name == "":
        raise TypeError("The collection_name parameter must be a string")

    if not isinstance(points, list):
        raise TypeError("The parameter points must be a list of PointStruct objects")
    else:
        if not all([isinstance(x, PointStruct) for x in points]):
            raise ValueError(
                "points cannot be an empty list and each entry must be a PointStruct object"
            )

    document_title = points[0].payload["metadata"]["title"]

    if document_in_collection(collection_name, document_title):
        raise ValueError(
            f"The points of the document {document_title} has been previously uploaded in"
            f"the collection {collection_name}. If you want to update them please use the"
            " update_points function"
        )

    client.upsert(
        collection_name=collection_name,
        wait=True,
        points=points,
    )
    logger.info(f"Points uploaded into the collection {collection_name}")


def update_points(collection_name: str, points: list[PointStruct]) -> None:
    """
    Update the information of points already uploaded into a vector database collection

    Args:
        collection_name: str -> Name of the vector db colleciton
        points: list[PointStruct] -> List of PointStruct objects, each entry of the list is a chunk of a document

    Return:
        None
    """
    logger.info("Updating points...")
    # Error handlers for parameters
    if not isinstance(collection_name, str) or collection_name == "":
        raise TypeError("The collection_name parameter must be a string")

    if not isinstance(points, list):
        raise TypeError("The parameter points must be a list of PointStruct objects")
    else:
        if not all([isinstance(x, PointStruct) for x in points]):
            raise ValueError(
                "points cannot be an empty list and each entry must be a PointStruct object"
            )

    document_title = points[0].payload["metadata"]["title"]

    if not document_in_collection(collection_name, document_title):
        logger.info(
            f"The document {document_title} has not been uploaded in the collection before"
        )

    else:
        # Delete all the points related to the previous uploaded document
        delete_document(collection_name, document_title)

    # In both cases, when the points has not been uploaded before, and when the points has been uplodaded,
    # It is needed to upload the points again
    upload_points(collection_name, points)


def create_collection(collection_name: str, vector_size: int) -> None:
    """
    Creates a collection if does not previously exists

    Args:
        collection_name: str -> Name of the collection that will store all the vectors
        vector_size: int -> Dimension of the vectors that will be stored.

    Return:
        None
    """
    logger.info("Creating collection...")
    # Error handlers for inputs
    if not isinstance(collection_name, str) or not isinstance(vector_size, int):
        raise TypeError(
            "The parameter collection_name must be a string, and vector_size must be an integer"
        )

    if vector_size <= 0:
        raise ValueError("vector_size must be greater than 1")

    # Check that the collection has not been created before
    if client.collection_exists(collection_name):
        logger.info("The collection already exists")
        return

    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=vector_size, distance=Distance.DOT),
    )
    logger.info("Collection created")


def create_points(
    chunks: list[dict],
) -> list[PointStruct]:
    """
    From the chunks created (list of dictionaries), create a list of PointStruct objects ready to be indexed into the Qdrant vector database

    Args:
        chunks: list[dict] -> list of Dictionaries, where each dictionary is a chunk. Each dictionary contains the keys:
                            'id' -> Id of the PointStruct, is a uuid string
                            'vector' -> vector of n dimensions
                            'payload' -> dictionary with two keys: "text" and "metadata"
    Returns:
        list[PointStruct] -> Returns a list of PointStruct, which is ready to be indexed into the vector database
    """
    # Mandatory keys to be present on each dictionary of the chunks list
    mandatory_keys = ["id", "vector", "payload"]

    logger.info("Creating points...")

    # Error handlers for chunks
    if not isinstance(chunks, list):
        raise TypeError("The 'chunks' parameter must be a list of dictionaries")

    # I assume that the chunks has the same structure
    if not all([x in chunks[0].keys() for x in mandatory_keys]):
        raise ValueError(
            f"All the chunks must contains the following keys: {', '.join(mandatory_keys)}"
        )

    # Create a list of PointStruct objects, each PointStruct object is a chunk
    points = [
        PointStruct(
            id=chunk_info["id"],
            vector=chunk_info["vector"],
            payload=chunk_info["payload"],
        )
        for chunk_info in chunks
    ]

    logger.info("Points created")

    return points
