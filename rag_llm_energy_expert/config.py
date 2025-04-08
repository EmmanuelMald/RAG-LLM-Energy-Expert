from pydantic import SecretStr
from pydantic_settings import BaseSettings


class GCP_CONFIG(BaseSettings):
    PROJECT_ID: str = "learned-stone-454021-c8"
    DEV_SA: SecretStr = SecretStr("mock_dev_service_account")
    BUCKET_NAME: str = "rag_llm_energy_expert"
    REGION: str = "northamerica-south1"


class LLM_CONFIG(BaseSettings):
    OPENAI_SECRET_ID: SecretStr = SecretStr("mock_openai_secret_id")
    OPENAI_VERSION_ID: SecretStr = SecretStr("mock_openai_version_id")
    LLAMA_SECRET_ID: SecretStr = SecretStr("mock_llama_secret_id")
    LLAMA_VERSION_ID: SecretStr = SecretStr("mock_llama_version_id")


class QDRANT_CONFIG(BaseSettings):
    URL: str = (
        "https://6bc62d49-364d-4a8b-82b5-9908cbb26d4e.us-east4-0.gcp.cloud.qdrant.io"
    )
    SECRET_ID: str = "QDRANT-KEY"
    VERSION_ID: str = "1"
    API_KEY: SecretStr = SecretStr("")
    COLLECTION_NAME: str = "energy_expert_"
    COLLECTION_VERSION: str = "v1"
    CHUNK_OVERLAP: int = 0
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
