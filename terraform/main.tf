provider "google" {
  project = var.gcp_project_id
  region  = var.gcp_region
  zone    = var.gcp_zone
}

############### ARTIFACT REGISTRY ###############

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


############### CLOUD RUN - EMBEDDING SERVICE ###############

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


############### BIGQUERY ###############
resource "google_bigquery_dataset" "energy_expert_dataset" {
  dataset_id                  = var.dataset_id
  description                 = "The datasets in Bigquery can be considered as schemas in any other structured database. So this is the schema for the tables."
  location                    = var.gcp_region
  default_table_expiration_ms = 3600000

  labels = {
    env = "default"
  }
}

resource "google_bigquery_table" "users_table" {
  dataset_id = google_bigquery_dataset.energy_expert_dataset.dataset_id
  table_id   = var.users_table_id

  labels = {
    env         = "default"
    primary_key = "user_id"
  }

  schema = <<EOF

[
  {
    "name": "user_id",
    "type": "STRING",
    "mode": "REQUIRED",
    "description": "User identifier"
  },
  {
    "name": "company_id",
    "type": "STRING",
    "mode": "REQUIRED",
    "description": "Company where the user works"
  },
  {
    "name": "created_at",
    "type": "TIMESTAMP",
    "mode": "REQUIRED",
    "description": "Timestamp when the user was created"
  },
  {
   "name": "full_name",
    "type": "STRING",
    "mode": "REQUIRED",
    "description": "Full name of the user"
  },
  {
    "name": "email",
    "type": "STRING",
    "mode": "NULLABLE",
    "description": "Email of the user"
  },
  {
    "name": "role",
    "type": "STRING",
    "mode": "NULLABLE",
    "description": "Role of the user in the company"
  }
]
EOF

}




resource "google_bigquery_table" "chat_history_table" {
  dataset_id = google_bigquery_dataset.energy_expert_dataset.dataset_id
  table_id   = var.chat_history_table_id

  labels = {
    env = "default"
  }

  schema = <<EOF

[
  {
    "name": "user_id",
    "type": "STRING",
    "mode": "REQUIRED",
    "description": "User identifier"
  },
  {
    "name": "chat_id",
    "type": "STRING",
    "mode": "REQUIRED",
    "description": "Company where the user works"
  },
  {
    "name": "created_at",
    "type": "TIMESTAMP",
    "mode": "REQUIRED",
    "description": "Timestamp when the user was created"
  },
  {
   "name": "prompt",
    "type": "STRING",
    "mode": "REQUIRED",
    "description": "Full name of the user"
  },
  {
    "name": "llm_response",
    "type": "STRING",
    "mode": "NULLABLE",
    "description": "Email of the user"
  },
  {
    "name": "llm_model",
    "type": "STRING",
    "mode": "NULLABLE",
    "description": "Role of the user in the company"
  }
]
EOF
}
