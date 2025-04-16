# Embedding Service

The embedding service converts a raw text into vectors. Generating embeddings is important because they let a computer understand the meaning of words or sentences in a smart way. Instead of just matching exact words, it can find things that are similar in meaning. This helps when searching for the best answers from documents. It makes the AI more accurate and useful.

This service does two main steps:

- chunk a raw text
- embed the chunked text into vectors

To make this work, both steps rely on an [*embedding model*](https://huggingface.co/blog/getting-started-with-embeddings):

- **chunking**: The input text is split into smaller pieces (chunks), based on how the embedding model processes and tokenizes text. Chunking is necessary because embedding models can only handle a limited number of tokens at a time.

- **embedding**: Each chunk is then converted into a vector — a set of numbers that captures the meaning of the text — which can later be used for searching or generating responses.

There's a [HuggingFace dashboard](https://huggingface.co/spaces/mteb/leaderboard) that compares the performance of different embedding models. Some metrics to focus on are:

- *Number of Parameters*: 

    A higher value means the model requires more CPU/GPU memory to run

- *Embedding Dimension*:

    The dimension of the vectors produced

- *Max tokens*:

    How many tokens the model can process, the higher the better.


For this time, we'll be using the [*sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2*](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2) model:

- Number of Parameters: 118M
- Embedding Dimension: 384
- Max tokens: 512

## Deployment

This service is deployed as a containerized application using FastAPI, Docker, Terraform, and Google Cloud Run. The deployment process is automated with a simple CI/CD pipeline that triggers on changes within the embeddings/ folder.

### Technologies Used
***FastAPI***: Provides a lightweight and high-performance API to expose the embedding functionality.

***Docker***: Containerizes the application for consistent execution across environments.

***Terraform***: Manages the cloud infrastructure, including the provisioning of Cloud Run and Artifact Registry.

***Cloud Run***: Hosts the service as a scalable, serverless container in Google Cloud.

***CI/CD (Cloud Build)***: A GitHub-triggered pipeline automatically builds, pushes, and redeploys the service when relevant changes are detected.

CI/CD Behavior:

Whenever a file is modified inside the embeddings/ folder (e.g., model logic, dependencies, etc.), the CI/CD workflow:

- Builds a new Docker image.

- Pushes the image to Artifact Registry.

- Deploys the new version to Cloud Run using Terraform.

This allows for rapid iteration and easy updates to the embedding service without manual intervention.