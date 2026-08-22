variable "aws_region" {
  type    = string
  default = "us-east-1"
}

variable "project_name" {
  type    = string
  default = "trialrag"
}

variable "github_repo" {
  description = "owner/repo, for the OIDC deploy role's trust policy"
  type        = string
  default     = "brannon-white/Rag-pipeline"
}

variable "max_daily_spend_usd" {
  description = "AWS Budgets alert threshold -- matches the app's own TRIALRAG_MAX_DAILY_SPEND_USD circuit breaker in spirit, but this one covers total AWS spend, not just Anthropic tokens."
  type        = number
  default     = 40
}

variable "budget_alert_email" {
  description = "Where the AWS Budgets alert notification goes."
  type        = string
}

# --- Secrets: sourced from TF_VAR_* environment variables at apply time,
# never written to a committed .tfvars file. Terraform creates the secret
# *shell* with these as the initial value; see modules/secrets for why
# ignore_changes then keeps subsequent applies from being a vector to leak
# or overwrite them, and terraform/README.md for the actual apply command.
variable "anthropic_api_key" {
  type      = string
  sensitive = true
}

variable "voyage_api_key" {
  type      = string
  sensitive = true
}

variable "database_url" {
  description = "Production Neon connection string."
  type        = string
  sensitive   = true
}

variable "initial_image_tag" {
  description = "ECR tag to deploy on the very first apply, before deploy.yml has ever run. See terraform/README.md Stage C -- an image must exist at this tag before App Runner can be created."
  type        = string
  default     = "bootstrap"
}
