# To check data types: https://developer.hashicorp.com/terraform/language/expressions/types

variable "gcp_project_id" {
    type = string
    description = "GCP project id"
    default = "learned-stone-454021-c8"
}

variable "gcp_region"{
    type = string
    description = "GCP region where the resources will be stored"
    default = "northamerica-south1"
}

variable "gcp_zone"{
    type = string
    description = "GCP zone within the gcp_region"
    default = "northamerica-south1-a"
}

variable "artifact_registry_name" {
    type = string
    description = "Name of the artifact registry to create"
    default = "rag-llm-artifact"
}

variable "env_prefix" {
    type = string
    description = "Prefix related to the environment in use"
}

variable "artifact_registry_dry_run"{
    type = bool
    description = "Determines if cleanup policies delete artifacts. true: No artifacts are deleted. false: Artifacts are deleted or kept depending on the policies"
    default = false
}

variable "cloudrun_embeddings_instance_name"{
    type = string
    description = "Name of the CloudRun instance for the embedding service"
    default = "embedding-service"
}

variable "cloudrun_embeddings_instance_port"{
    type = number
    description = "Port where the container will listen"
    default = 8080
}