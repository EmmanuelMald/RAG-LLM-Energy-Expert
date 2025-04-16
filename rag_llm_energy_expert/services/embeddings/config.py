from pydantic_settings import BaseSettings


class EmbeddingsConfig(BaseSettings):
    CHUNK_OVERLAP: int = 0
    EMBEDDING_MODEL: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
