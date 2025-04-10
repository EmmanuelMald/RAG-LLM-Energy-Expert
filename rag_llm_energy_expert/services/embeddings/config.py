from pydantic_settings import BaseSettings


class EMBEDDINGS_CONFIG(BaseSettings):
    CHUNK_OVERLAP: int = 10
    EMBEDDING_MODEL: str = "nomic-ai/nomic-embed-text-v2-moe"
