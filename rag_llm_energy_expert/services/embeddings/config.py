from pydantic_settings import BaseSettings


class EmbeddingsConfig(BaseSettings):
    CHUNK_OVERLAP: int = 10
    EMBEDDING_MODEL: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
