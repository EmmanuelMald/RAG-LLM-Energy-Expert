# Ingestion Pipeline

This module is responsible for processing and transforming PDF documents into embeddings, and then store them in a vector database. It is a core component of the RAG-LLM Energy Expert system.

## Steps

The ingestion pipeline performs the following steps:

1. **Document Parsing**: Reads a file (currently only supports PDF files), stored either in the local device or from files stored on Google Cloud Storage.

2. **Embeddings Generation**: Uses the [embedding service](../embeddings) deployed on CloudRun to chunk and embed the obtained pdf text in the step 1.

3. **Vector Store Insertion**: Stores the embeddings into a vector database for efficient semantic search. In this case, the embeddings are stored in the [Qdrant VectorDB](https://try.qdrant.tech/high-performance-vector-search?utm_source=google&utm_medium=cpc&utm_campaign=21518712216&utm_content=163351119817&utm_term=quadrant%20vector%20db&hsa_acc=6907203950&hsa_cam=21518712216&hsa_grp=163351119817&hsa_ad=724496064473&hsa_src=g&hsa_tgt=kwd-2276315971848&hsa_kw=quadrant%20vector%20db&hsa_mt=e&hsa_net=adwords&hsa_ver=3&gad_source=1&gbraid=0AAAAAodw_9BwA2DNo0CcxnxWkrGXPYJJt&gclid=Cj0KCQjwqv2_BhC0ARIsAFb5Ac9v90NfWkGLPKdumd33GE8CdAVmMEE0FnFmjbPI2wI9fW9TQXgV35saAj73EALw_wcB)
