GCP_PROJECT_ID="learned-stone-454021-c8"
GCP_SA="dev-service-account@learned-stone-454021-c8.iam.gserviceaccount.com"
GCP_REGION="northamerica-south1"

gcloud-auth:
	gcloud config unset auth/impersonate_service_account
	gcloud auth application-default login --impersonate-service-account $(GCP_SA)
	
sync-dependencies:
	uv sync --all-groups