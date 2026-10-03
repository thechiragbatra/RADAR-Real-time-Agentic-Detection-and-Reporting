output "scorer_url" {
  value = "http://${aws_lb.this.dns_name}"
}

output "artefact_bucket" {
  value = aws_s3_bucket.artefacts.bucket
}

output "model_registry_uri" {
  value = local.model_uri
}

output "kinesis_stream" {
  value = aws_kinesis_stream.transactions.name
}

output "investigation_queue_url" {
  value = aws_sqs_queue.investigations.url
}

output "alerts_topic_arn" {
  value = aws_sns_topic.alerts.arn
}

output "ecr_scorer" {
  value = aws_ecr_repository.scorer.repository_url
}

output "ecr_lambda" {
  value = aws_ecr_repository.lambda.repository_url
}

output "state_table" {
  value = aws_dynamodb_table.state.name
}

output "dashboard_url" {
  value = "https://${var.aws_region}.console.aws.amazon.com/cloudwatch/home?region=${var.aws_region}#dashboards:name=${aws_cloudwatch_dashboard.main.dashboard_name}"
}

output "ecs_cluster" {
  value = aws_ecs_cluster.this.name
}
