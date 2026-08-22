locals {
  name = var.project_name
}

module "ecr" {
  source          = "./modules/ecr"
  repository_name = local.name
}

module "s3" {
  source      = "./modules/s3"
  bucket_name = "${local.name}-artifacts-${data.aws_caller_identity.current.account_id}"
}

module "secrets" {
  source            = "./modules/secrets"
  name_prefix       = local.name
  anthropic_api_key = var.anthropic_api_key
  voyage_api_key    = var.voyage_api_key
  database_url      = var.database_url
}

module "iam" {
  source       = "./modules/iam"
  project_name = local.name
  github_repo  = var.github_repo
  secret_arns = [
    module.secrets.anthropic_api_key_arn,
    module.secrets.voyage_api_key_arn,
    module.secrets.database_url_arn,
  ]
  ecr_repository_arn = module.ecr.repository_arn
}

module "apprunner" {
  source                       = "./modules/apprunner"
  service_name                 = local.name
  ecr_repository_url           = module.ecr.repository_url
  initial_image_tag            = var.initial_image_tag
  ecr_access_role_arn          = module.iam.apprunner_ecr_access_role_arn
  instance_role_arn            = module.iam.apprunner_instance_role_arn
  database_url_secret_arn      = module.secrets.database_url_arn
  anthropic_api_key_secret_arn = module.secrets.anthropic_api_key_arn
  voyage_api_key_secret_arn    = module.secrets.voyage_api_key_arn
}

module "observability" {
  source                 = "./modules/observability"
  project_name           = local.name
  apprunner_service_name = local.name
  apprunner_service_id   = module.apprunner.service_id
  max_daily_spend_usd    = var.max_daily_spend_usd
  budget_alert_email     = var.budget_alert_email
}

data "aws_caller_identity" "current" {}
