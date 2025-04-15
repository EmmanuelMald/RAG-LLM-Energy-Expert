from google.cloud.iam_credentials_v1 import IAMCredentialsClient

import sys

sys.path.append("..")

from rag_llm_energy_expert.config import QdrantConfig, GCPConfig
from utils.gcp.secret_manager import get_secret

gcp_config = GCPConfig()


def get_qdrant_config() -> QdrantConfig:
    """
    Get the qdrant client with secret information

    Args:
        None

    Return:
        QdrantConfig instance
    """
    qdrant = QdrantConfig()

    # Get secret id and version id
    secret_id = qdrant.SECRET_ID
    version_id = qdrant.VERSION_ID

    # Get api_key
    api_key = get_secret(secret_id, version_id, gcp_config.PROJECT_ID)

    return QdrantConfig(API_KEY=api_key)


def generate_id_token(audience: str) -> str:
    """
    To access CloudRun and other GCP services, all the endpoints require an authorization ID Token.
    The ID token can be obtained from a service account that is authorized to use the service.

    Args:
        audience:  Indicates who will receive or verify the token

    Return:
        str -> ID Token
    """
    cred_client = IAMCredentialsClient()

    name = f"projects/-/serviceAccounts/{gcp_config.DEV_SA.get_secret_value()}"

    response_token = cred_client.generate_id_token(name=name, audience=audience)

    return response_token.token
