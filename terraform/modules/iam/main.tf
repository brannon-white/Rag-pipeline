variable "project_name" {
  type = string
}

variable "github_repo" {
  description = "owner/repo, e.g. brannon-white/Rag-pipeline"
  type        = string
}

variable "secret_arns" {
  description = "The 3 Secrets Manager ARNs the running App Runner instance needs to read."
  type        = list(string)
}

variable "ecr_repository_arn" {
  type = string
}

# ---------------------------------------------------------------------------
# GitHub OIDC: no long-lived AWS keys stored in GitHub secrets for deploy.yml.
# Thumbprint is fetched live rather than hardcoded -- GitHub has rotated its
# CA before, and a stale hardcoded thumbprint is a silent future outage.
# ---------------------------------------------------------------------------

data "tls_certificate" "github" {
  url = "https://token.actions.githubusercontent.com/.well-known/openid-configuration"
}

resource "aws_iam_openid_connect_provider" "github" {
  url             = "https://token.actions.githubusercontent.com"
  client_id_list  = ["sts.amazonaws.com"]
  thumbprint_list = [data.tls_certificate.github.certificates[0].sha1_fingerprint]
}

# Scoped to `main` -- both deploy.yml's triggers (push and workflow_dispatch)
# resolve to the same `ref:refs/heads/main` subject when run against that
# branch, so one condition value covers both.
#
# sts:TagSession is required here, not just AssumeRoleWithWebIdentity --
# aws-actions/configure-aws-credentials tags the assumed session with GitHub
# context by default (repo, actor, workflow, ...) unless
# `role-skip-session-tagging: true` is set. Without this action allowed, AWS
# rejects the *entire* AssumeRoleWithWebIdentity call, surfacing only as a
# generic "Not authorized to perform sts:AssumeRoleWithWebIdentity" with no
# mention of tagging anywhere in the error -- confirmed live, this is exactly
# what broke the first real deploy.yml run.
data "aws_iam_policy_document" "github_deploy_trust" {
  statement {
    effect  = "Allow"
    actions = ["sts:AssumeRoleWithWebIdentity", "sts:TagSession"]

    principals {
      type        = "Federated"
      identifiers = [aws_iam_openid_connect_provider.github.arn]
    }

    condition {
      test     = "StringEquals"
      variable = "token.actions.githubusercontent.com:aud"
      values   = ["sts.amazonaws.com"]
    }

    condition {
      test     = "StringEquals"
      variable = "token.actions.githubusercontent.com:sub"
      values   = ["repo:${var.github_repo}:ref:refs/heads/main"]
    }
  }
}

resource "aws_iam_role" "github_deploy" {
  name               = "${var.project_name}-github-deploy"
  assume_role_policy = data.aws_iam_policy_document.github_deploy_trust.json
}

resource "aws_iam_role_policy_attachment" "github_deploy_ecr" {
  role       = aws_iam_role.github_deploy.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryPowerUser"
}

data "aws_iam_policy_document" "github_deploy_apprunner" {
  statement {
    effect = "Allow"
    actions = [
      "apprunner:StartDeployment",
      "apprunner:DescribeService",
      "apprunner:UpdateService",
      "apprunner:ListServices",
    ]
    resources = ["*"] # App Runner deployment actions don't support resource-level scoping
  }
}

resource "aws_iam_role_policy" "github_deploy_apprunner" {
  name   = "${var.project_name}-apprunner-deploy"
  role   = aws_iam_role.github_deploy.id
  policy = data.aws_iam_policy_document.github_deploy_apprunner.json
}

# ---------------------------------------------------------------------------
# App Runner's two distinct roles: one it assumes to *pull the image*
# (build-time), one the *running service* assumes to read secrets
# (task-time). Conflating them is a common App Runner setup mistake --
# they have different trust principals entirely.
# ---------------------------------------------------------------------------

data "aws_iam_policy_document" "apprunner_build_trust" {
  statement {
    effect  = "Allow"
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["build.apprunner.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "apprunner_ecr_access" {
  name               = "${var.project_name}-apprunner-ecr-access"
  assume_role_policy = data.aws_iam_policy_document.apprunner_build_trust.json
}

resource "aws_iam_role_policy_attachment" "apprunner_ecr_access" {
  role       = aws_iam_role.apprunner_ecr_access.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSAppRunnerServicePolicyForECRAccess"
}

data "aws_iam_policy_document" "apprunner_instance_trust" {
  statement {
    effect  = "Allow"
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["tasks.apprunner.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "apprunner_instance" {
  name               = "${var.project_name}-apprunner-instance"
  assume_role_policy = data.aws_iam_policy_document.apprunner_instance_trust.json
}

data "aws_iam_policy_document" "apprunner_instance_secrets" {
  statement {
    effect    = "Allow"
    actions   = ["secretsmanager:GetSecretValue"]
    resources = var.secret_arns
  }
}

resource "aws_iam_role_policy" "apprunner_instance_secrets" {
  name   = "${var.project_name}-read-secrets"
  role   = aws_iam_role.apprunner_instance.id
  policy = data.aws_iam_policy_document.apprunner_instance_secrets.json
}

output "github_deploy_role_arn" {
  value = aws_iam_role.github_deploy.arn
}

output "apprunner_ecr_access_role_arn" {
  value = aws_iam_role.apprunner_ecr_access.arn
}

output "apprunner_instance_role_arn" {
  value = aws_iam_role.apprunner_instance.arn
}
