from pydantic_settings import BaseSettings


class EmbeddingsConfig(BaseSettings):
    CHUNK_OVERLAP: int = 100
    EMBEDDING_MODEL: str = "sentence-transformers/LaBSE"
