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
    EMBED_TEXT_ENDPOINT: str = "/embed-text"
    EMBEDDING_SERVICE_IDTOKEN: SecretStr = ""
    BQ_DATASET: str = "energy_expert"
    BQ_CHAT_HISTORY_TABLE: str = "chat_history"
    BQ_USERS_TABLE: str = "users"


class QdrantConfig(BaseSettings):
    URL: str = (
        "https://6bc62d49-364d-4a8b-82b5-9908cbb26d4e.us-east4-0.gcp.cloud.qdrant.io"
    )
    SECRET_ID: str = "QDRANT-KEY"
    VERSION_ID: str = "1"
    API_KEY: SecretStr = ""
    COLLECTION_NAME: str = "energy_expert_"
    COLLECTION_VERSION: str = "v1"
    DOCUMENTS_RETRIEVED_LIMIT: int = 5


class LLMConfig(BaseSettings):
    SECRET_ID: str = "GEMINI-API-KEY"
    SECRET_VERSION: str = "1"
    API_KEY: SecretStr = ""
    MODEL: str = "gemini-2.0-flash"
