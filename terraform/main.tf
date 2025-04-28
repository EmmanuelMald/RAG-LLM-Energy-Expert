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
  dataset_id  = var.dataset_id
  description = "The datasets in Bigquery can be considered as schemas in any other structured database. So this is the schema for the tables."
  location    = var.gcp_region

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
    "name": "company_name",
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
    "name": "company_role",
    "type": "STRING",
    "mode": "NULLABLE",
    "description": "Role of the user in the company"
  },
  {
    "name": "last_entered_at",
    "type": "TIMESTAMP",
    "mode": "REQUIRED",
    "description": "Last time the user entered the chatbot"
  }
]
EOF
}




resource "google_bigquery_table" "prompts_table" {
  dataset_id = google_bigquery_dataset.energy_expert_dataset.dataset_id
  table_id   = var.prompts_table_id

  labels = {
    env = "default"
  }

  schema = <<EOF

[
  
  {
    "name": "prompt_id",
    "type": "STRING",
    "mode": "REQUIRED",
    "description": "Id of the chat history"
  },
  {
    "name": "session_id",
    "type": "STRING",
    "mode": "REQUIRED",
    "description": "Id of the session which the prompt belongs to"
  },
  {
    "name": "prompt",
    "type": "STRING",
    "mode": "REQUIRED",
    "description": "User's prompt"
  },
  {
    "name": "context",
    "type": "STRING",
    "mode": "NULLABLE",
    "description": "Context of the prompt"
  },
  {
    "name": "llm_response",
    "type": "STRING",
    "mode": "REQUIRED",
    "description": "Response from the LLM"
  },
  {
   "name": "prompt_created_at",
    "type": "TIMESTAMP",
    "mode": "REQUIRED",
    "description": "Timestamp when the prompt was created"
  },
  {
    "name": "context_retrieved_at",
    "type": "TIMESTAMP",
    "mode": "REQUIRED",
    "description": "Timestamp when the context was retrieved"
  },
  {
    "name": "llm_response_created_at",
    "type": "TIMESTAMP",
    "mode": "REQUIRED",
    "description": "Timestamp when the LLM response was created"
  },
  {
    "name": "documents_retrieved",
    "type": "INTEGER",
    "mode": "REQUIRED",
    "description": "Number of documents retrieved to build context"
  },
  {
    "name": "temperature",
    "type": "FLOAT",
    "mode": "REQUIRED",
    "description": "Temperature used in the LLM response"
  }
]
EOF
}




resource "google_bigquery_table" "chat_sessions_table" {
  dataset_id = google_bigquery_dataset.energy_expert_dataset.dataset_id
  table_id   = var.chat_sessions_table_id

  labels = {
    env = "default"
  }

  schema = <<EOF

[
  {
    "name": "session_id",
    "type": "STRING",
    "mode": "REQUIRED",
    "description": "Id of the chat session"
  },
  {
    "name": "llm_version_id",
    "type": "STRING",
    "mode": "REQUIRED",
    "description": "Id of the LLM used"
  },
  {
    "name": "user_id",
    "type": "STRING",
    "mode": "REQUIRED",
    "description": "Id of the user who started the chat session"
  },
  {
    "name": "created_at",
    "type": "TIMESTAMP",
    "mode": "REQUIRED",
    "description": "Timestamp when the chat session was created"
  },
  {
    "name": "last_used_at",
    "type": "TIMESTAMP",
    "mode": "REQUIRED",
    "description": "Id of the LLM used"
  },
  {
    "name": "session_history",
    "type": "JSON",
    "mode": "REQUIRED",
    "description": "Chat history of the session. To be introduced to the LLM as history context"
  }
]
EOF
}




resource "google_bigquery_table" "llms_table" {
  dataset_id = google_bigquery_dataset.energy_expert_dataset.dataset_id
  table_id   = var.llms_table_id

  labels = {
    env = "default"
  }

  schema = <<EOF

[
  {
    "name": "llm_version_id",
    "type": "STRING",
    "mode": "REQUIRED",
    "description": "Id of the LLM used"
  },
  {
    "name": "llm_model_name",
    "type": "STRING",
    "mode": "REQUIRED",
    "description": "Model of the LLM used"
  },
  {
    "name": "system_prompt",
    "type": "STRING",
    "mode": "REQUIRED",
    "description": "System prompt of the LLM"
  },
  {
    "name": "temperature",
    "type": "FLOAT",
    "mode": "REQUIRED",
    "description": "Temperature of the LLM"
  },
  {
    "name": "created_at",
    "type": "TIMESTAMP",
    "mode": "REQUIRED",
    "description": "Timestamp when the LLM was configured by the first time"
  },
  {
    "name": "last_used_at",
    "type": "TIMESTAMP",
    "mode": "NULLABLE",
    "description": "Timestamp when the LLM was used for the last time"
  },
  {
    "name": "last_user_id",
    "type": "STRING",
    "mode": "NULLABLE",
    "description": "Id of the user who used the LLM for the last time"
  }
]
EOF
}
