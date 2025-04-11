from fastapi import FastAPI
from models import EmbeddingRequest, EmbeddingResponse, Chunk

import sys

sys.path.append("..")

from embedding_pipeline import text_embedder

app = FastAPI()


@app.post("/embed-text", response_model=EmbeddingResponse)
def generate_embeddings(request: EmbeddingRequest):
    raw_chunks = text_embedder(
        text=request.text,
        chunk_overlap=request.chunk_overlap,
        embedding_model_name=request.embedding_model_name,
        metadata=request.metadata,
    )

    list_of_chunks = [
        Chunk(
            vector_id=chunk["vector_id"],
            vector=chunk["vector"],
            metadata=chunk["metadata"],
        )
        for chunk in raw_chunks
    ]

    response = EmbeddingResponse(list_of_chunks)

    return response
