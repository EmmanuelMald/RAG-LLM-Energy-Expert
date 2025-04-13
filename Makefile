GCP_PROJECT_ID="learned-stone-454021-c8"
GCP_SA="dev-service-account@learned-stone-454021-c8.iam.gserviceaccount.com"
GCP_REGION="northamerica-south1"

gcloud-auth:
	gcloud config unset auth/impersonate_service_account
	gcloud auth application-default login --impersonate-service-account $(GCP_SA)
	
uv-sync:
	uv sync --all-groups

install-git-hooks: 
	pre-commit install

# For CloudBuild, it is necessary to being executed in the us-central1 region
# check https://cloud.google.com/build/docs/locations#restricted_regions_for_some_projects
run-embedding-service-cicd:
	gcloud builds submit --region=us-central1 --config embedding_service.yaml