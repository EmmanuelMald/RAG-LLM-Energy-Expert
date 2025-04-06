from pydantic import SecretStr
import sys

sys.path.append("..")

from rag_llm_energy_expert.config import QDRANT_CONFIG
from utils.gcp.secret_manager import get_secret


def get_qdrant_config() -> QDRANT_CONFIG:
    """
    Get the qdrant client with secret information

    Args:
        None

    Return:
        QDRANT_CONFIG instance
    """
    # Get secret id and version id
    secret_id = QDRANT_CONFIG.SECRET_ID
    version_id = QDRANT_CONFIG.VERSION_ID

    # Get api_key
    api_key = SecretStr(get_secret(secret_id, version_id))

    return QDRANT_CONFIG(API_KEY=api_key)
