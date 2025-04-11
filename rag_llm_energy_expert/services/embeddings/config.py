from pydantic_settings import BaseSettings


class EmbeddingsConfig(BaseSettings):
    CHUNK_OVERLAP: int = 10
    EMBEDDING_MODEL: str = "nomic-ai/nomic-embed-text-v2-moe"
