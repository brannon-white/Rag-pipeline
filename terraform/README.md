# TrialRAG infrastructure

Terraform-managed AWS deployment: App Runner, ECR, S3 (artifacts), Secrets
Manager, IAM (GitHub OIDC), CloudWatch, AWS Budgets. See the root plan
document's M4 section for the full architecture rationale (App Runner vs
Lambda, cost table, etc).

## One-time bootstrap (already done for this project — documented for a
## future rebuild, not something to re-run)

Terraform can't manage the S3 bucket holding its own state, so this part is
plain `aws` CLI, run once, before `terraform init` ever runs:

```bash
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
BUCKET="trialrag-tfstate-${ACCOUNT_ID}"

aws s3api create-bucket --bucket "$BUCKET" --region us-east-1
aws s3api put-bucket-versioning --bucket "$BUCKET" \
  --versioning-configuration Status=Enabled
aws s3api put-bucket-encryption --bucket "$BUCKET" \
  --server-side-encryption-configuration \
  '{"Rules":[{"ApplyServerSideEncryptionByDefault":{"SSEAlgorithm":"AES256"}}]}'
aws s3api put-public-access-block --bucket "$BUCKET" \
  --public-access-block-configuration \
  BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true
```

If the bucket name changes, update the literal `bucket` value in
`backend.tf` to match — backend blocks can't reference variables.

## First apply

Requires an image already pushed to ECR at the `initial_image_tag` tag
(default `"bootstrap"`) — App Runner's `aws_apprunner_service` resource fails
to create if the referenced image doesn't exist yet, and nothing has ever
been pushed on a brand-new repository:

```bash
terraform init

# 1. Apply just the ECR repository first, so there's somewhere to push to.
terraform apply -target=module.ecr

# 2. Push a real image at the bootstrap tag.
aws ecr get-login-password --region us-east-1 | \
  docker login --username AWS --password-stdin "$(terraform output -raw ecr_repository_url | cut -d/ -f1)"
docker buildx build --platform linux/amd64 -t "$(terraform output -raw ecr_repository_url):bootstrap" ..
docker push "$(terraform output -raw ecr_repository_url):bootstrap"

# 3. Now the rest can be created, since the image it needs already exists.
#    App Runner service creation takes ~5 minutes -- this is normal, not stuck.
TF_VAR_anthropic_api_key="$(grep '^ANTHROPIC_API_KEY=' ../.env | cut -d= -f2-)" \
TF_VAR_voyage_api_key="$(grep '^VOYAGE_API_KEY=' ../.env | cut -d= -f2-)" \
TF_VAR_database_url="<production Neon connection string>" \
terraform apply
```

**Expect the `observability` module's two `aws_cloudwatch_log_group` resources
to fail on this first apply** with `ResourceAlreadyExistsException` — App
Runner creates its own `/aws/apprunner/<service>/<service-id>/{application,service}`
log groups the moment the service starts, before Terraform ever gets to them.
Everything else (App Runner, IAM, secrets, S3) will have succeeded; fix just
the log groups by importing what App Runner already made, then re-apply to
set the actual retention policy on them:

```bash
SERVICE_ID=$(terraform state show module.apprunner.aws_apprunner_service.this | grep -oE '[0-9a-f]{32}')
terraform import module.observability.aws_cloudwatch_log_group.apprunner_application \
  "/aws/apprunner/trialrag/${SERVICE_ID}/application"
terraform import module.observability.aws_cloudwatch_log_group.apprunner_service \
  "/aws/apprunner/trialrag/${SERVICE_ID}/service"
terraform apply   # now just sets retention_in_days on both, 0 -> 14
```

## Rotating a secret

`modules/secrets` sets `ignore_changes` on each secret's value, so a routine
`terraform apply` will never touch it again after the first one. Rotate for
real with:

```bash
aws secretsmanager put-secret-value \
  --secret-id trialrag/anthropic-api-key \
  --secret-string "<new value>"
```

Then restart the App Runner service (or just wait for the next deploy) so it
picks up the new value — App Runner reads `runtime_environment_secrets` at
deployment time, not continuously.

## Every deploy after the first

Handled by `.github/workflows/deploy.yml` via OIDC (`github_deploy_role_arn`
output above) — not by re-running `terraform apply`. Terraform only re-enters
the picture for infrastructure changes (new module, resource config change),
never for a routine code deploy.
