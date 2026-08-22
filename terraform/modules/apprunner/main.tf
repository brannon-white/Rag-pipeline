variable "service_name" {
  type = string
}

variable "ecr_repository_url" {
  type = string
}

variable "initial_image_tag" {
  type = string
}

variable "ecr_access_role_arn" {
  type = string
}

variable "instance_role_arn" {
  type = string
}

variable "database_url_secret_arn" {
  type = string
}

variable "anthropic_api_key_secret_arn" {
  type = string
}

variable "voyage_api_key_secret_arn" {
  type = string
}

resource "aws_apprunner_auto_scaling_configuration_version" "this" {
  auto_scaling_configuration_name = "${var.service_name}-scaling"
  min_size                        = 1
  max_size                        = 3
}

resource "aws_apprunner_service" "this" {
  service_name = var.service_name

  source_configuration {
    authentication_configuration {
      access_role_arn = var.ecr_access_role_arn
    }

    # image_identifier only changes here on the very first apply
    # (initial_image_tag = "bootstrap"). Every deploy after that is done by
    # deploy.yml calling `aws apprunner update-service` directly with the
    # new git-SHA tag -- not by re-running Terraform -- so this resource's
    # image_identifier is intentionally left to drift from what's actually
    # running; see `ignore_changes` below.
    image_repository {
      image_identifier      = "${var.ecr_repository_url}:${var.initial_image_tag}"
      image_repository_type = "ECR"

      image_configuration {
        port = "8000"

        runtime_environment_variables = {
          TRIALRAG_ENV = "prod"
        }

        runtime_environment_secrets = {
          TRIALRAG_DATABASE_URL = var.database_url_secret_arn
          ANTHROPIC_API_KEY     = var.anthropic_api_key_secret_arn
          VOYAGE_API_KEY        = var.voyage_api_key_secret_arn
        }
      }
    }

    # CI deploys explicit SHA tags via `update-service`; App Runner must
    # never auto-redeploy on its own image-scan/push triggers.
    auto_deployments_enabled = false
  }

  instance_configuration {
    cpu               = "1024" # 1 vCPU
    memory            = "2048" # 2 GB
    instance_role_arn = var.instance_role_arn
  }

  health_check_configuration {
    protocol            = "HTTP"
    path                = "/healthz"
    interval            = 10
    timeout             = 5
    healthy_threshold   = 1
    unhealthy_threshold = 3
  }

  auto_scaling_configuration_arn = aws_apprunner_auto_scaling_configuration_version.this.arn

  lifecycle {
    ignore_changes = [source_configuration[0].image_repository[0].image_identifier]
  }
}

output "service_url" {
  value = "https://${aws_apprunner_service.this.service_url}"
}

output "service_arn" {
  value = aws_apprunner_service.this.arn
}

output "service_id" {
  value = aws_apprunner_service.this.service_id
}
