from pydantic_settings import BaseSettings


class EmbeddingsConfig(BaseSettings):
    CHUNK_OVERLAP: int = 10
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
