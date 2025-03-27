import pymupdf
import pymupdf4llm
import sys
from loguru import logger

sys.path.append("..")

from gcp_utils.gcs import get_file, blob_exists
from rag_llm_energy_expert.config import GCP_CONFIG

# Initialize the config values for GCP
config = GCP_CONFIG()


def extract_pdf_content(
    gcs_file_path: str,
    bucket_name: str = config.BUCKET_NAME,
) -> str:
    """
    Parse a pdf that is stored in Google Cloud Storage (GCS)

    Args:
        gcs_file_path: str -> Path to the file in gcs. Ex: "gcs_folder/file.pdf"
        bucket_path: str -> Name of the gcs bucket. Ex: "my_bucket"

    Return:
        str -> String with all the pdf parsed in a markdown format
    """
    # blob_exists already has error handlers for the parameters
    if not blob_exists(gcs_file_path, bucket_name):
        raise ValueError(f"The file {gcs_file_path} does not exists")

    extension = gcs_file_path.split(".")[-1]

    if extension != "pdf":
        raise TypeError("The file is not a pdf")

    title = gcs_file_path.split("/")[-1].split(".")[0]

    # Load the file in memory
    logger.info("Loading file from GCS...")
    pdf_file_bytes = get_file(gcs_file_path, bucket_name)

    # Create a Document object that is like a list, where
    # each entry is a page
    logger.info("Extracting PDF content...")
    file = pymupdf.Document(stream=pdf_file_bytes)

    # Reads the PDF with its metadata and creates a list of dictionaries
    logger.info("Converting to markdown format...")
    md_text = pymupdf4llm.to_markdown(
        file,
        # page_chunks = True, # Create a list of pages of the Document 
        # extract_words=True, # Adds key words to each page dictionary
        show_progress = False,
    )

    file_data = {
        "text": md_text,
        "title": title,
        "gcs_path": gcs_file_path,
    }

    return file_data


def chunk_pdf_content(
        file_data: str, 
        chunk_size: int, 
        overlap: int,
        ) -> list[str]:
    """
    Chunks the PDF content into chunks
    """
