from pydantic import BaseModel, Field
from typing import Optional

import sys

sys.path.append("..")

from config import EmbeddingsConfig

embeddings_config = EmbeddingsConfig()

embedding_model = embeddings_config.EMBEDDING_MODEL
embedding_model_description = f"""Name of the embedding model. 
Must be available on SentenceTransformers. By default, the embedding model is {embedding_model}
"""


class TextRequest(BaseModel):
    text: str = Field(
        description="String with the whole text to be embedded", min_length=1
    )
    embedding_model_name: str = Field(
        default=embedding_model, min_length=1, description=embedding_model_description
    )
    chunk_overlap: Optional[int] = Field(default=embeddings_config.CHUNK_OVERLAP, ge=0)
    metadata: Optional[dict[str, str]] = Field(
        default=None,
        description="Data associated to the text. Ex: {'title':'title_name', 'author':'author_name'}",
    )


class TextResponse(BaseModel):
    embedded_chunks: list[dict]
