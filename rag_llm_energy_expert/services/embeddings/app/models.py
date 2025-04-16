from pydantic import BaseModel, Field, field_validator
from typing import Optional, Union

import sys

sys.path.append("..")

from config import EmbeddingsConfig

embeddings_config = EmbeddingsConfig()


class Payload(BaseModel):
    text: str = Field(description="Text of the chunk that was embedded in a vector")
    metadata: Optional[Union[dict[str, str], None]] = Field(
        default=None, description="Dictionary with the metadata provided"
    )


class Chunk(BaseModel):
    vector_id: str = Field(
        description="uuid string representing the id of the vector created."
    )
    vector: list[float] = Field(
        description="vector representing the text of the chunk."
    )
    payload: Payload


class EmbeddingRequest(BaseModel):
    text: str = Field(
        description="String with the whole text to be embedded", min_length=1
    )
    embedding_model_name: Optional[Union[str, None]] = Field(
        default=embeddings_config.EMBEDDING_MODEL,
        min_length=1,
        description="Name of the embedding model. Must be available on SentenceTransformers.",
    )
    chunk_overlap: Optional[Union[int, None]] = Field(
        default=embeddings_config.CHUNK_OVERLAP, ge=0
    )
    metadata: Optional[Union[dict[str, str], None]] = Field(
        default=None,
        description="Data associated to the text. Ex: {'title':'title_name', 'date':'9999-12-23'}.",
    )

    @field_validator("embedding_model_name", mode="after")
    @classmethod
    def validate_embedding_model_name(cls, value):
        if value is None:
            return embeddings_config.EMBEDDING_MODEL
        return value

    @field_validator("chunk_overlap", mode="after")
    @classmethod
    def validate_chunk_overlap(cls, value):
        if value is None:
            return embeddings_config.CHUNK_OVERLAP
        return value


class EmbeddingResponse(BaseModel):
    chunks: list[Chunk]
