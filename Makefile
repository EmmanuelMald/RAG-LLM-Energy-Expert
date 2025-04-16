GCP_PROJECT_ID="learned-stone-454021-c8"
GCP_SA="dev-service-account@learned-stone-454021-c8.iam.gserviceaccount.com"
GCP_REGION="northamerica-south1"
EMBEDDING_SERVICE_IMAGE_NAME="northamerica-south1-docker.pkg.dev/$(GCP_PROJECT_ID)/dev-rag-llm-artifact/embedding_service:latest"


gcloud-auth:
	gcloud config unset auth/impersonate_service_account 
	gcloud auth application-default login --impersonate-service-account $(GCP_SA)
	
uv-sync:
	uv sync --all-groups

install-git-hooks: 
	uv run pre-commit install
	uv run pre-commit install-hooks

################################# EMBEDDING SERVICE ###################################
run-embedding-service-api:
	cd rag_llm_energy_expert/services/embeddings/app &&\
	uv run -- uvicorn main:app --reload 

build-embedding-service-container:
	cp pyproject.toml uv.lock rag_llm_energy_expert/services/embeddings/.
	docker build \
	-f rag_llm_energy_expert/services/embeddings/embeddings_service.dockerfile \
	-t $(EMBEDDING_SERVICE_IMAGE_NAME) \
	rag_llm_energy_expert/services/embeddings
	rm rag_llm_energy_expert/services/embeddings/pyproject.toml \
	rag_llm_energy_expert/services/embeddings/uv.lock 

run-embedding-service-container:
	docker run -p 8080:8080 $(EMBEDDING_SERVICE_IMAGE_NAME)

# For CloudBuild, it is necessary to being executed in the us-central1 region
# check https://cloud.google.com/build/docs/locations#restricted_regions_for_some_projects
run-embedding-service-cicd:
	gcloud builds submit --region=us-central1 --config embedding_service.yaml