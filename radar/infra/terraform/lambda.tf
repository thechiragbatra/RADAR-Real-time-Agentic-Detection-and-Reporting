locals {
  lambda_env = {
    RADAR_ENV                     = var.env
    RADAR_AWS_REGION              = var.aws_region
    RADAR_MODEL_URI               = local.model_uri
    RADAR_ARTEFACT_BUCKET         = aws_s3_bucket.artefacts.bucket
    RADAR_USER_STATE_TABLE        = aws_dynamodb_table.state.name
    RADAR_REFERENCE_TABLE         = aws_dynamodb_table.state.name
    RADAR_DECISIONS_TABLE         = aws_dynamodb_table.decisions.name
    RADAR_CASES_TABLE             = aws_dynamodb_table.cases.name
    RADAR_INVESTIGATION_QUEUE_URL = aws_sqs_queue.investigations.url
    RADAR_ALERTS_TOPIC_ARN        = aws_sns_topic.alerts.arn
    RADAR_FIREHOSE_FEATURE_LOG    = aws_kinesis_firehose_delivery_stream.feature_log.name
    RADAR_SCORER_URL              = "http://${aws_lb.this.dns_name}"
    RADAR_SCORER_API_KEY          = var.scorer_api_key
    RADAR_BEDROCK_REGION          = var.bedrock_region
    RADAR_BEDROCK_MODEL_ID        = var.bedrock_model_id
    RADAR_PSI_ALERT_THRESHOLD     = tostring(var.psi_alert_threshold)
  }
}

resource "aws_cloudwatch_log_group" "lambda" {
  for_each          = toset(["stream-consumer", "investigate", "drift-monitor"])
  name              = "/aws/lambda/${local.name}-${each.key}"
  retention_in_days = 14
}

# ---- Stream consumer ------------------------------------------------------------------------
resource "aws_lambda_function" "stream_consumer" {
  function_name = "${local.name}-stream-consumer"
  role          = aws_iam_role.lambda.arn
  package_type  = "Image"
  image_uri     = local.lambda_image
  timeout       = 60
  memory_size   = 1024
  architectures = ["x86_64"]

  image_config {
    command = ["radar.lambdas.stream_consumer.handler"]
  }

  environment {
    variables = local.lambda_env
  }

  depends_on = [aws_cloudwatch_log_group.lambda]
}

resource "aws_lambda_event_source_mapping" "kinesis" {
  event_source_arn                   = aws_kinesis_stream.transactions.arn
  function_name                      = aws_lambda_function.stream_consumer.arn
  starting_position                  = "LATEST"
  batch_size                         = 100
  maximum_batching_window_in_seconds = 1
  parallelization_factor             = 1 # one invocation per shard keeps per-user ordering
  bisect_batch_on_function_error     = true
  maximum_retry_attempts             = 3
  function_response_types            = ["ReportBatchItemFailures"]

  destination_config {
    on_failure {
      destination_arn = aws_sqs_queue.consumer_failures.arn
    }
  }
}

# ---- Investigator ---------------------------------------------------------------------------------
resource "aws_lambda_function" "investigate" {
  function_name                  = "${local.name}-investigate"
  role                           = aws_iam_role.lambda.arn
  package_type                   = "Image"
  image_uri                      = local.lambda_image
  timeout                        = 300
  memory_size                    = 2048
  architectures                  = ["x86_64"]
  reserved_concurrent_executions = 5 # Bedrock throughput is the bottleneck; let alerts queue

  image_config {
    command = ["radar.lambdas.investigate.handler"]
  }

  environment {
    variables = local.lambda_env
  }

  depends_on = [aws_cloudwatch_log_group.lambda]
}

resource "aws_lambda_event_source_mapping" "sqs" {
  event_source_arn        = aws_sqs_queue.investigations.arn
  function_name           = aws_lambda_function.investigate.arn
  batch_size              = 1
  function_response_types = ["ReportBatchItemFailures"]
}

# ---- Drift monitor (hourly) -------------------------------------------------------------------
resource "aws_lambda_function" "drift_monitor" {
  function_name = "${local.name}-drift-monitor"
  role          = aws_iam_role.lambda.arn
  package_type  = "Image"
  image_uri     = local.lambda_image
  timeout       = 300
  memory_size   = 2048
  architectures = ["x86_64"]

  image_config {
    command = ["radar.lambdas.drift_monitor.handler"]
  }

  environment {
    variables = local.lambda_env
  }

  depends_on = [aws_cloudwatch_log_group.lambda]
}

resource "aws_cloudwatch_event_rule" "drift_schedule" {
  name                = "${local.name}-drift-hourly"
  schedule_expression = "rate(1 hour)"
}

resource "aws_cloudwatch_event_target" "drift_schedule" {
  rule = aws_cloudwatch_event_rule.drift_schedule.name
  arn  = aws_lambda_function.drift_monitor.arn
}

resource "aws_lambda_permission" "drift_schedule" {
  statement_id  = "AllowEventBridgeInvoke"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.drift_monitor.function_name
  principal     = "events.amazonaws.com"
  source_arn    = aws_cloudwatch_event_rule.drift_schedule.arn
}

# ---- DriftDetected -> retrain task; plus a weekly scheduled retrain --------------------
resource "aws_cloudwatch_event_rule" "drift_detected" {
  name = "${local.name}-drift-detected"
  event_pattern = jsonencode({
    source        = ["radar.drift"]
    "detail-type" = ["DriftDetected"]
  })
}

resource "aws_cloudwatch_event_rule" "weekly_retrain" {
  name                = "${local.name}-weekly-retrain"
  schedule_expression = "cron(0 2 ? * MON *)"
}

resource "aws_cloudwatch_event_target" "retrain" {
  for_each = {
    drift  = aws_cloudwatch_event_rule.drift_detected.name
    weekly = aws_cloudwatch_event_rule.weekly_retrain.name
  }
  rule     = each.value
  arn      = aws_ecs_cluster.this.arn
  role_arn = aws_iam_role.events.arn

  ecs_target {
    task_definition_arn = aws_ecs_task_definition.retrain.arn
    launch_type         = "FARGATE"
    task_count          = 1
    network_configuration {
      subnets          = aws_subnet.public[*].id
      security_groups  = [aws_security_group.scorer.id]
      assign_public_ip = true
    }
  }
}
