import pymupdf
import pymupdf4llm
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from datetime import datetime
import sys
from loguru import logger

sys.path.append("..")

from gcp_utils.gcs import get_file, blob_exists



def extract_pdf_content(
    gcs_file_path: str,
    bucket_name: str,
) -> dict[str, str]:
    """
    Parse a pdf that is stored in Google Cloud Storage (GCS)

    Args:
        gcs_file_path: str -> Path to the file in gcs. Ex: "gcs_folder/file.pdf"
        bucket_path: str -> Name of the gcs bucket. Ex: "my_bucket"

    Return:
        file_data: dict[str, str] -> Dictionary with all the pdf parsed in a markdown format and metadata
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

    logger.info("PDF content successfully extracted")

    return file_data


def chunk_by_md_headers(
        md_text: str,
        markdown_headers_to_split_on: list[tuple[str]] = [("#", 'Header 1'), ("##", "Header 2"), ("###", "Header 3"), ("####", "Header 4"), ("#####", "Header 5")],
) -> list:
    """
    Split the data by the markdown headers
    
    Args:
        md_text: str -> Text in Markdown format
        markdown_headers_to_split_on: list[tuple[str]] -> List of tuples, each entry is a tuple containing the header to split on

    Returns:
        list[Document] -> List of Documents, each Document has 'page_content' and 'metadata' attributes
    """
    logger.info("Chunking the text by the markdown headers...")

    if not isinstance(md_text, str):
        raise TypeError("The md_text parameter must be a markdown formatted string")
    
    if not isinstance(markdown_headers_to_split_on, list):
        raise TypeError("The markdown_headers_to_split_on parameter must be a list of tuples")
    
    else:
        if not all(isinstance(x, tuple) for x in markdown_headers_to_split_on):
            raise TypeError("All the entries of the markdown_headers_to_split_on must be a tuple of strings")
    
    # Initialize a MarkdonHeaderTextSplitter object
    markdown_splitter = MarkdownHeaderTextSplitter(markdown_headers_to_split_on, strip_headers=False) 

    # Split the text by the headers
    md_headers_chunks = markdown_splitter.split_text(md_text)

    logger.info("Chunks based on markdown headers successfully created")

    return md_headers_chunks


def size_md_chunks(md_headers_chunks: list, chunk_size: int, chunk_overlap: int = 0,) -> list:
    """
    Once the text has been chunked by the markdown headers, its time to split each chunk even more based on the
    embedding model specifics.

    Args:
        md_headers_chunks: list[Document] -> List of chunks (Documents) splitter by markdown headers
        chunk_size: int -> Number of tokens to split the md chunks
        chunk_overlap: int -> Number of tokens that will be overlapped on each chunk
    Returns:
        chunks_sized: list[Document] -> List of chunks (Documents) splitted by the chunk size
    """
    logger.info("Sizing markdown chunks...")

    if not isinstance(md_headers_chunks, list):
        raise TypeError("The md_headers_chunks parameter must be a list of Documents")
    if not all(isinstance(x, int) for x in [chunk_size, chunk_overlap]):
        raise TypeError("chunk_size and chunk_overlap must be integers")
    

    text_splitter = RecursiveCharacterTextSplitter(chunk_size = chunk_size, chunk_overlap = chunk_overlap)

    # Returns a list of Documents, but now each chunk has the specific size and overlap
    chunks_sized = text_splitter.split_documents(md_headers_chunks) 

    logger.info("Markdown chunks sized successfully")

    return chunks_sized
    

def prepare_chunks_for_embeddings(
        chunks_sized: list,
        file_data: dict[str, str],
) -> list[dict]:
    """
    Create the final chunks that will be embedded as vectors.

    Args:
        chunks_sized: list[Documents] -> List of Documents sized
        file_data: dict[str, str] -> Dictionary with the next keys:
                                - title: str -> Name of the document
                                - gcs_path: str -> The GCS path where its stored. ex: "gs://folder/text.pdf"
    
    Returns:
        list[dict] -> List of dictionaries, each dictionary is a chunk
    """
    mandatory_file_data_keys = ["title", "gcs_path"]

    logger.info("Preparing chunks for embedding...")

    if not isinstance(chunks_sized, list):
        raise TypeError("The chunks_sized parameter must be a list of Documents")
    
    if not isinstance(file_data, dict):
        raise TypeError("file_data parameter must be a dictionary")
    
    else:
        if not all([x in file_data.keys() for x in mandatory_file_data_keys]):
            raise ValueError(f"file_data parameter must contain the next keys: {mandatory_file_data_keys}")
    
    logger.info("Creating additional metadata")
    extra_metadata = {
        "upload_date": datetime.now().strftime(r"%Y-%m%d %H:%M:%S"),
        "title": file_data["title"],
        "storage_path": file_data["gcs_path"],
    }

    # Creating a list of dictionaries, each entry of the list is the metadata of each chunk
    logger.info("Extracting chunk's metadata")
    final_chunks = [doc.metadata for doc in chunks_sized]

    logger.info("Creating chunks to embed...")
    for i, final_chunk in enumerate(final_chunks):

        # Add the page content in the metadata
        final_chunk["data"] = chunks_sized[i].page_content

        # Add extra_metadata to each chunk_metadata
        final_chunk.update(extra_metadata)
    
    logger.info("Chunks ready to be embedded")

    return final_chunks
