import sys

sys.path.append("..")

from rag_llm_energy_expert.config import QDRANT_CONFIG, GCPConfig
from utils.gcp.secret_manager import get_secret

gcp_config = GCPConfig()


def get_qdrant_config() -> QDRANT_CONFIG:
    """
    Get the qdrant client with secret information

    Args:
        None

    Return:
        QDRANT_CONFIG instance
    """
    qdrant = QDRANT_CONFIG()

    # Get secret id and version id
    secret_id = qdrant.SECRET_ID
    version_id = qdrant.VERSION_ID

    # Get api_key
    api_key = get_secret(secret_id, version_id, gcp_config.PROJECT_ID)

    return QDRANT_CONFIG(API_KEY=api_key)
