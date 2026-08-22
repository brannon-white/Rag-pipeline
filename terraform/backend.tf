# Backend config can't reference variables (Terraform must resolve it before
# any variable evaluation happens), so the bucket name is a literal here --
# it must match whatever Stage B's one-time bootstrap created. See
# terraform/README.md for that bootstrap sequence; it deliberately isn't a
# Terraform resource, since Terraform can't manage the bucket holding its own
# state without a chicken-and-egg problem.
terraform {
  required_version = ">= 1.9"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    tls = {
      source  = "hashicorp/tls"
      version = "~> 4.0"
    }
  }

  backend "s3" {
    bucket       = "trialrag-tfstate-928967253230"
    key          = "trialrag/terraform.tfstate"
    region       = "us-east-1"
    use_lockfile = true
    encrypt      = true
  }
}

provider "aws" {
  region = var.aws_region
}
