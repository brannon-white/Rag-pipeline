variable "repository_name" {
  type = string
}

resource "aws_ecr_repository" "this" {
  name = var.repository_name

  # MUTABLE, not IMMUTABLE: tried IMMUTABLE first on the reasoning that git
  # SHA tags never intentionally get reused, but confirmed live it doesn't
  # coexist with how BuildKit actually pushes -- BuildKit's registry client
  # can resend the final manifest PUT as a defensive retry even after a
  # push already succeeded (observed live: log shows "pushing manifest ...
  # done" immediately followed by "tag already exists" for a tag that had
  # genuinely never been pushed before that same job). Against a mutable
  # tag that redundant retry is a harmless no-op; against IMMUTABLE it's a
  # hard failure on every single push. Content-addressing by git SHA is
  # still the real invariant here regardless of this setting -- mutability
  # only controls whether the *registry* would also enforce it, and
  # deploy.yml already guarantees every tag maps to exactly one commit by
  # construction.
  image_tag_mutability = "MUTABLE"

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
