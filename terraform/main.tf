provider "google" {
  project = var.gcp_project_id
  region  = var.gcp_region
  zone    = var.gcp_zone
}

resource "google_artifact_registry_repository" "rag_llm_artifact_registry" {
  location               = var.gcp_region
  repository_id          = "${var.env_prefix}-${var.artifact_registry_name}"
  format                 = "docker"
  cleanup_policy_dry_run = var.artifact_registry_dry_run
  cleanup_policies {
    id     = "delete_untagged_images"
    action = "DELETE"
    condition {
      tag_state  = "UNTAGGED"
      older_than = "10d" # after 10 days untagged, delete the image 
    }
  }
}

resource "google_cloud_run_v2_service" "cloudrun_embeddings_instance" {
  name                = var.cloudrun_embeddings_instance_name
  location            = var.gcp_region
  client              = "terraform"
  deletion_protection = false

  template {
    containers {
      image = "${var.gcp_region}-docker.pkg.dev/${var.gcp_project_id}/${var.env_prefix}-${var.artifact_registry_name}/embedding_service:${var.embedding_service_tag_image}"
      ports {
        container_port = var.cloudrun_embeddings_instance_port
      }
      resources {
        limits = {
          memory = "4Gi"
          cpu    = "1"
        }
      }
    }
    scaling {
      # Min instances
      min_instance_count = 1
      max_instance_count = 2
    }


  }
}

resource "google_cloud_run_v2_service_iam_member" "auth" {
  location = google_cloud_run_v2_service.cloudrun_embeddings_instance.location
  name     = google_cloud_run_v2_service.cloudrun_embeddings_instance.name
  role     = "roles/run.invoker"
  member   = "serviceAccount:${var.gcp_dev_sa}"
}
