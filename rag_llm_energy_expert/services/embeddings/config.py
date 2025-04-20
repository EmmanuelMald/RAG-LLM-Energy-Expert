from pydantic_settings import BaseSettings


class EmbeddingsConfig(BaseSettings):
    CHUNK_OVERLAP: int = 100
    EMBEDDING_MODEL: str = "sentence-transformers/multi-qa-MiniLM-L6-cos-v1"
