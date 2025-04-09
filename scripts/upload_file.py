import argparse
import sys

sys.path.append("..")

from rag_llm_energy_expert.ingest_document import upload_file
from rag_llm_energy_expert.config import QDRANT_CONFIG


def main():
    # Load config values
    qdrant_config = QDRANT_CONFIG()

    # Create parser
    parser = argparse.ArgumentParser(
        description="This script loads a raw PDF stored either in Google Cloud Storage or in the local into a vector DB"
    )

    # Add args
    parser.add_argument(
        "-f",
        "--file_path",
        required=True,
        help="Either a gcs path: (ex: 'gs://bucket_name/folder_name/pdf_name.pdf') or a local path (can be a relative path or a full path ex: 'local_folder/pdf_file.pdf' or 'C:Users/folder/pdf_file.pdf')",
    )

    parser.add_argument(
        "--chunk-overlap",
        required=False,
        help="Number of tokens to overlap between chunks.",
        default=qdrant_config.CHUNK_OVERLAP,
    )

    parser.add_argument(
        "-c",
        "--vectordb-collection",
        required=False,
        help="Name of the vector DB collection where the document's chunks will be stored.",
        default=qdrant_config.COLLECTION_NAME + qdrant_config.COLLECTION_VERSION,
    )

    parser.add_argument(
        "-m",
        "--embedding-model",
        required=False,
        help="Name of the embedding model to use. Must be available on sentence-transformers.",
        default=qdrant_config.EMBEDDING_MODEL,
    )

    # Parse args
    args = parser.parse_args()

    upload_file(
        file_path=args.file_path,
        embedding_model=args.embedding_model,
        chunk_overlap=args.chunk_overlap,
        collection_name=args.vectordb_collection,
    )


if __name__ == "__main__":
    main()
