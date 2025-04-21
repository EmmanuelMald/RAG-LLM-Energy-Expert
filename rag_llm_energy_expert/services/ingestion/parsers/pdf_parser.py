import pymupdf
import pymupdf4llm
from datetime import datetime
from loguru import logger
from typing import Union
import os

import sys

sys.path.append("../../../..")

from rag_llm_energy_expert.utils.gcp.gcs import get_file


def parse_pdf_file(
    pdf_path: str,
) -> dict[str, Union[str | dict]]:
    """
    Parse a pdf that is stored in Google Cloud Storage (GCS) or in the local

    Args:
        pdf_path: str -> Either a gcs path: (ex: 'gs://bucket_name/folder_name/pdf_name.pdf') or
                        a local path (can be a relative path or a full path ex: 'local_folder/pdf_file.pdf' or 'C:Users/folder/pdf_file.pdf')

    Return:
        file_data: Union[str | dict] -> Dictionary with all the text parsed and metadata, it has the format:
                                            {"text": "string with all the PDF text", "metadata": {"key": "value"}}
    """
    logger.info("Extracting PDF content...")
    # Datatype Check
    if not isinstance(pdf_path, str):
        raise ValueError(
            "pdf_path parameter must be a string containing either a gcs path (ex: 'gs://bucket_name/folder_name/pdf_name.pdf')"
            " or a local path (can be a relative path or a full path ex: 'local_folder/pdf_file.pdf' or 'C:Users/folder/pdf_file.pdf')"
        )

    # File extension check
    file_extension = pdf_path.split(".")[-1]

    if file_extension != "pdf":
        raise TypeError("The file is not a pdf")

    # Checking if the path comes from gcs or is a local path
    if pdf_path.startswith("gs://"):
        logger.info("Google cloud storage path detected")
        # Taking out the 'gs://' part of the string
        useful_pdf_path = pdf_path[5:]

        # Splitting the pdf path in two
        pdf_path_parts = useful_pdf_path.split("/", maxsplit=1)

        try:
            bucket_name = pdf_path_parts[0]
            blob_name = pdf_path_parts[1]

        except Exception as e:
            raise ValueError(
                "There is an error in the pdf_path parameter. if you try to set a gcs file path,"
                f" use the following format: 'gs://bucket_name/path/to/file.pdf'. {e}"
            )

        # Download in memory the pdf from GCS
        pdf_bytes = get_file(gcs_file_path=blob_name, bucket_name=bucket_name)

        # Load the PDF content into a Document object, each entry of the Document is a page
        pdf_document = pymupdf.Document(stream=pdf_bytes)

    # If the path seems to be a local path
    else:
        logger.info("Local path detected")
        if not os.path.isfile(pdf_path):
            raise ValueError(f"The file {pdf_path} does not exists")

        # load the pdf into a Document object, each entry of the Document is a page
        pdf_document = pymupdf.Document(pdf_path)

    file_title = pdf_path.split("/")[-1].split(".")[0]

    pdf_text = pymupdf4llm.to_markdown(pdf_document)

    file_data = {
        "text": pdf_text,
        "metadata": {
            "title": file_title,
            "storage_path": pdf_path,
            "upload_date": datetime.now().strftime(r"%Y-%m-%d"),
        },
    }

    logger.info("PDF content successfully extracted")

    return file_data
