from pydantic import SecretStr
from pydantic_settings import BaseSettings


class GCPConfig(BaseSettings):
    PROJECT_ID: str = "learned-stone-454021-c8"
    DEV_SA: str = "dev-service-account@learned-stone-454021-c8.iam.gserviceaccount.com"
    BUCKET_NAME: str = "rag_llm_energy_expert"
    REGION: str = "northamerica-south1"
    EMBEDDING_SERVICE_URL: str = (
        "https://embedding-service-214571216460.northamerica-south1.run.app"
    )


class QdrantConfig(BaseSettings):
    URL: str = (
        "https://6bc62d49-364d-4a8b-82b5-9908cbb26d4e.us-east4-0.gcp.cloud.qdrant.io"
    )
    SECRET_ID: str = "QDRANT-KEY"
    VERSION_ID: str = "1"
    API_KEY: SecretStr = SecretStr("")
    COLLECTION_NAME: str = "energy_expert_"
    COLLECTION_VERSION: str = "v1"
    CHUNK_OVERLAP: int = 10
    EMBEDDING_MODEL: str = "nomic-ai/nomic-embed-text-v2-moe"
