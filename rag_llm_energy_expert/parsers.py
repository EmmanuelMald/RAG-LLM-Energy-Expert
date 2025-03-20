import pymupdf
import sys
from loguru import logger

sys.path.append("..")

from gcp_utils.gcs import get_file, blob_exists
from rag_llm_energy_expert.config import GCP_CONFIG

# Initialize the config values for GCP
config = GCP_CONFIG()


def parse_pdf(
    gcs_file_path: str,
    bucket_name: str = config.BUCKET_NAME,
) -> str:
    """
    Parse a pdf that is stored in Google Cloud Storage (GCS)

    Args:
        gcs_file_path: str -> Path to the file in gcs. Ex: "gcs_folder/file.pdf"
        bucket_path: str -> Name of the gcs bucket. Ex: "my_bucket"

    Return:
        str -> String with all the pdf parsed
    """
    # blob_exists already has error handlers for the parameters
    if not blob_exists(gcs_file_path, bucket_name):
        raise ValueError(f"The file {gcs_file_path} does not exists")

    extension = gcs_file_path.split(".")[-1]

    if extension != "pdf":
        raise TypeError("The file is not a pdf")

    # Load the file in memory
    pdf_file_bytes = get_file(gcs_file_path, bucket_name)

    # Create a Document object that is like a list, where
    # each entry is a page
    file = pymupdf.Document(stream=pdf_file_bytes)

    # Get all the content of the pdf
    file_data = "".join([page.get_text() for page in file])

    return file_data
