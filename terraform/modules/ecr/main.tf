variable "repository_name" {
  type = string
}

resource "aws_ecr_repository" "this" {
  name = var.repository_name

  # IMMUTABLE because deploy.yml tags every image with the git SHA and
  # never reuses a tag -- if a tag *could* be overwritten, "which commit is
  # actually running" would stop being a question Terraform state or the
  # image tag alone could answer.
  image_tag_mutability = "IMMUTABLE"

  image_scanning_configuration {
    scan_on_push = true
  }
}

resource "aws_ecr_lifecycle_policy" "keep_last_10" {
  repository = aws_ecr_repository.this.name
  policy = jsonencode({
    rules = [{
      rulePriority = 1
      description  = "Keep only the last 10 images"
      selection = {
        tagStatus   = "any"
        countType   = "imageCountMoreThan"
        countNumber = 10
      }
      action = { type = "expire" }
    }]
  })
}

output "repository_url" {
  value = aws_ecr_repository.this.repository_url
}

output "repository_arn" {
  value = aws_ecr_repository.this.arn
}

output "repository_name" {
  value = aws_ecr_repository.this.name
}
