import argparse
import sys

sys.path.append("..")

from rag_llm_energy_expert.ingest_document import upload_document
from rag_llm_energy_expert.config import GCP_CONFIG, QDRANT_CONFIG


def main():
    # Load config values
    gcp_config = GCP_CONFIG()
    qdrant_config = QDRANT_CONFIG()

    # Create parser
    parser = argparse.ArgumentParser(
        description="This script loads a raw PDF stored in Google Cloud Storage into a vector DB"
    )

    # Add args
    parser.add_argument(
        "-p",
        "--gcs-document-path",
        required=True,
        help="Path inside the GCS bucket. Ex: 'gcs_folder1/gcs_folder2/pdf_name.pdf'",
    )

    parser.add_argument(
        "-b",
        "--bucket-name",
        required=False,
        help="Name of the GCS bucket where the file is stored. Ex: 'bucket_name'",
        default=gcp_config.BUCKET_NAME,
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

    upload_document(
        gcs_document_path=args.gcs_document_path,
        bucket_name=args.bucket_name,
        vectordb_collection=args.vectordb_collection,
        embedding_model=args.embedding_model,
        chunk_overlap=args.chunk_overlap,
    )


if __name__ == "__main__":
    main()
