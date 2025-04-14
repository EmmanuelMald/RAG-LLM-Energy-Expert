from fastapi import FastAPI, HTTPException
from models import EmbeddingRequest, EmbeddingResponse, Chunk, Payload
from loguru import logger

import sys

sys.path.append("..")

from embedding_pipeline import text_embedder

app = FastAPI()


@app.post("/embed-text", response_model=EmbeddingResponse)
def generate_embeddings(request: EmbeddingRequest):
    try:
        raw_chunks = text_embedder(
            text=request.text,
            chunk_overlap=request.chunk_overlap,
            embedding_model_name=request.embedding_model_name,
            metadata=request.metadata,
        )

        list_of_chunks = [
            Chunk(
                vector_id=chunk["id"],
                vector=chunk["vector"],
                payload=Payload(
                    text=chunk["payload"]["text"],
                    metadata=chunk["payload"]["metadata"],
                ),
            )
            for chunk in raw_chunks
        ]

    except Exception as e:
        logger.error(e)
        raise HTTPException(status_code=500, detail=str(e))

    response = EmbeddingResponse(chunks=list_of_chunks)

    return response
