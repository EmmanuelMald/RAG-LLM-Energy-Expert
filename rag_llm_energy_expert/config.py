from pydantic import SecretStr
from pydantic_settings import BaseSettings


class GCP_CONFIG(BaseSettings):
    PROJECT_ID: str = "natural-reserve-454018-j9"
    DEV_SA: SecretStr = SecretStr("mock_dev_service_account")
    BUCKET_NAME: str = "rag_llm_energy_expert"
    REGION: str = "northamerica-south1"

class LLM_CONFIG(BaseSettings):
    OPENAI_API_KEY: SecretStr = SecretStr("mock_openai_api_key")