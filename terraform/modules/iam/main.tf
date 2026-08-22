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

locals {
  # Split once, reused by the sub wildcard below -- GitHub's real subject
  # claim inserts a numeric ID between each name and the delimiter that
  # follows it (owner@id/repo@id), so the wildcard has to sit right after
  # each bare name, not after the whole "owner/repo" string.
  github_owner = split("/", var.github_repo)[0]
  github_name  = split("/", var.github_repo)[1]
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

# Scoped to `main` via the separate `repository`/`ref` claims, not the
# composite `sub` string -- confirmed live (see the decoded token dumped by a
# temporary debug step) that GitHub now embeds numeric owner/repo IDs
# straight into `sub`, e.g. `repo:brannon-white@141594485/Rag-pipeline@1342258430:ref:...`,
# not the plain `repo:owner/repo:ref:...` every tutorial assumes. A trust
# policy built on that assumption never matches. `repository` and `ref` are
# separate top-level claims on the same token and don't have this problem;
# conditioning on those directly is also just more robust against any future
# `sub` format change, not merely a workaround for this one.
#
# sts:TagSession is required here too, not just AssumeRoleWithWebIdentity --
# aws-actions/configure-aws-credentials tags the assumed session with GitHub
# context by default (repo, actor, workflow, ...) unless
# `role-skip-session-tagging: true` is set. Without this action allowed, AWS
# rejects the *entire* AssumeRoleWithWebIdentity call with no mention of
# tagging anywhere in the error. Both this and the sub-format issue above
# were hit in the same real deploy.yml debugging session; both needed fixing
# before OIDC auth actually succeeded.
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
      variable = "token.actions.githubusercontent.com:repository"
      values   = [var.github_repo]
    }

    condition {
      test     = "StringEquals"
      variable = "token.actions.githubusercontent.com:ref"
      values   = ["refs/heads/main"]
    }

    # AWS's own IAM validation rejects a GitHub OIDC trust policy that
    # doesn't condition on `sub` (or `job_workflow_ref`) at all -- confirmed
    # live: dropping this in favor of the repository/ref conditions above
    # fails apply with "must evaluate ... sub ... which is not scoped to
    # all." The wildcard sits right after each bare name (not after the
    # whole "owner/repo" string) so it lines up with where GitHub actually
    # inserts the numeric ID -- a single trailing `*` would never match
    # "owner@id/repo@id" at all, only "owner/repo@id".
    condition {
      test     = "StringLike"
      variable = "token.actions.githubusercontent.com:sub"
      values   = ["repo:${local.github_owner}*/${local.github_name}*:ref:refs/heads/main"]
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
