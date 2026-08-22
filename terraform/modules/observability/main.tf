variable "project_name" {
  type = string
}

variable "apprunner_service_name" {
  type = string
}

variable "apprunner_service_id" {
  description = "App Runner appends this to its auto-created log group names -- not known until the service exists, hence the dependency on the apprunner module's output rather than a name we could construct ourselves up front."
  type        = string
}

variable "max_daily_spend_usd" {
  type = number
}

variable "budget_alert_email" {
  type = string
}

# App Runner creates these log groups itself on first deploy; Terraform only
# takes over their retention policy (cost control -- unbounded retention on
# a public endpoint's access logs is a slow-motion bill surprise).
resource "aws_cloudwatch_log_group" "apprunner_application" {
  name              = "/aws/apprunner/${var.apprunner_service_name}/${var.apprunner_service_id}/application"
  retention_in_days = 14
}

resource "aws_cloudwatch_log_group" "apprunner_service" {
  name              = "/aws/apprunner/${var.apprunner_service_name}/${var.apprunner_service_id}/service"
  retention_in_days = 14
}

resource "aws_budgets_budget" "monthly" {
  name         = "${var.project_name}-monthly"
  budget_type  = "COST"
  limit_amount = tostring(var.max_daily_spend_usd * 30)
  limit_unit   = "USD"
  time_unit    = "MONTHLY"

  notification {
    comparison_operator        = "GREATER_THAN"
    threshold                  = 80
    threshold_type             = "PERCENTAGE"
    notification_type          = "ACTUAL"
    subscriber_email_addresses = [var.budget_alert_email]
  }

  notification {
    comparison_operator        = "GREATER_THAN"
    threshold                  = 100
    threshold_type             = "PERCENTAGE"
    notification_type          = "FORECASTED"
    subscriber_email_addresses = [var.budget_alert_email]
  }
}
