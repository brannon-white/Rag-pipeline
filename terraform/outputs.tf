output "service_url" {
  description = "Live public URL of the deployed API."
  value       = module.apprunner.service_url
}

output "ecr_repository_url" {
  value = module.ecr.repository_url
}

output "github_deploy_role_arn" {
  description = "Paste into deploy.yml / the repo's GitHub Actions config as the OIDC role to assume."
  value       = module.iam.github_deploy_role_arn
}

output "artifacts_bucket_name" {
  value = module.s3.bucket_name
}
