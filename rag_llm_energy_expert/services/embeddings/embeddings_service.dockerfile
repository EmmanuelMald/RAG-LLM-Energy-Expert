# Pull a Python image
FROM python:3.11-slim

ENV UV_VERSION=0.5.4


# Create the main directoy where everything will be stored
WORKDIR /embeddings

# Copy pyproject.toml and uv.lock in the working directory
COPY pyproject.toml uv.lock ./

# Upgrade pip and install the required uv version
RUN pip install --upgrade pip &&\
    pip install uv==${UV_VERSION}

# Create a requirements.txt from the pyproject.toml
RUN uv export --group general-dev --group llm --no-hashes -o requirements.txt

RUN pip install -r requirements.txt

COPY app/.  ./app/

COPY config.py embedding_pipeline.py __init__.py ./

# Move to the app directory to execute uvicorn without errors
WORKDIR /embeddings/app/

# Specify the port where the app will listen
EXPOSE 5000

# Execute uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "5000"]


