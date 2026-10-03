# Alarms route to the same SNS topic as fraud alerts (email subscription optional).

resource "aws_cloudwatch_metric_alarm" "scorer_5xx" {
  alarm_name          = "${local.name}-scorer-5xx"
  namespace           = "AWS/ApplicationELB"
  metric_name         = "HTTPCode_Target_5XX_Count"
  statistic           = "Sum"
  period              = 60
  evaluation_periods  = 3
  threshold           = 5
  comparison_operator = "GreaterThanThreshold"
  treat_missing_data  = "notBreaching"
  dimensions = {
    LoadBalancer = aws_lb.this.arn_suffix
    TargetGroup  = aws_lb_target_group.scorer.arn_suffix
  }
  alarm_actions = [aws_sns_topic.alerts.arn]
}

resource "aws_cloudwatch_metric_alarm" "scorer_p95_latency" {
  alarm_name          = "${local.name}-scorer-p95-latency"
  namespace           = "AWS/ApplicationELB"
  metric_name         = "TargetResponseTime"
  extended_statistic  = "p95"
  period              = 60
  evaluation_periods  = 5
  threshold           = 0.1 # seconds; SLO is p95 < 100 ms end-to-end at the ALB
  comparison_operator = "GreaterThanThreshold"
  treat_missing_data  = "notBreaching"
  dimensions = {
    LoadBalancer = aws_lb.this.arn_suffix
    TargetGroup  = aws_lb_target_group.scorer.arn_suffix
  }
  alarm_actions = [aws_sns_topic.alerts.arn]
}

resource "aws_cloudwatch_metric_alarm" "consumer_iterator_age" {
  alarm_name          = "${local.name}-consumer-lag"
  namespace           = "AWS/Lambda"
  metric_name         = "IteratorAge"
  statistic           = "Maximum"
  period              = 60
  evaluation_periods  = 3
  threshold           = 60000 # ms behind the stream head
  comparison_operator = "GreaterThanThreshold"
  treat_missing_data  = "notBreaching"
  dimensions = {
    FunctionName = aws_lambda_function.stream_consumer.function_name
  }
  alarm_actions = [aws_sns_topic.alerts.arn]
}

resource "aws_cloudwatch_metric_alarm" "lambda_errors" {
  for_each = {
    stream-consumer = aws_lambda_function.stream_consumer.function_name
    investigate     = aws_lambda_function.investigate.function_name
    drift-monitor   = aws_lambda_function.drift_monitor.function_name
  }
  alarm_name          = "${local.name}-${each.key}-errors"
  namespace           = "AWS/Lambda"
  metric_name         = "Errors"
  statistic           = "Sum"
  period              = 300
  evaluation_periods  = 1
  threshold           = 3
  comparison_operator = "GreaterThanThreshold"
  treat_missing_data  = "notBreaching"
  dimensions = {
    FunctionName = each.value
  }
  alarm_actions = [aws_sns_topic.alerts.arn]
}

resource "aws_cloudwatch_metric_alarm" "investigations_dlq" {
  alarm_name          = "${local.name}-investigations-dlq"
  namespace           = "AWS/SQS"
  metric_name         = "ApproximateNumberOfMessagesVisible"
  statistic           = "Maximum"
  period              = 300
  evaluation_periods  = 1
  threshold           = 0
  comparison_operator = "GreaterThanThreshold"
  treat_missing_data  = "notBreaching"
  dimensions = {
    QueueName = aws_sqs_queue.investigations_dlq.name
  }
  alarm_actions = [aws_sns_topic.alerts.arn]
}

resource "aws_cloudwatch_metric_alarm" "drift" {
  alarm_name          = "${local.name}-feature-drift"
  namespace           = "RADAR"
  metric_name         = "MaxPSI"
  statistic           = "Maximum"
  period              = 3600
  evaluation_periods  = 1
  threshold           = var.psi_alert_threshold
  comparison_operator = "GreaterThanOrEqualToThreshold"
  treat_missing_data  = "notBreaching"
  alarm_actions       = [aws_sns_topic.alerts.arn]
}

resource "aws_cloudwatch_dashboard" "main" {
  dashboard_name = local.name
  dashboard_body = jsonencode({
    widgets = [
      {
        type = "metric", x = 0, y = 0, width = 12, height = 6
        properties = {
          title  = "Scoring throughput & flags (EMF)"
          region = var.aws_region
          stat   = "Sum", period = 60
          metrics = [
            ["RADAR", "ScoredTransactions", "ModelVersion", "ALL"],
            [".", "Flagged", ".", "."]
          ]
        }
      },
      {
        type = "metric", x = 12, y = 0, width = 12, height = 6
        properties = {
          title  = "Scorer latency (ALB p50/p95/p99)"
          region = var.aws_region
          period = 60
          metrics = [
            ["AWS/ApplicationELB", "TargetResponseTime", "LoadBalancer", aws_lb.this.arn_suffix, { stat = "p50" }],
            ["...", { stat = "p95" }],
            ["...", { stat = "p99" }]
          ]
        }
      },
      {
        type = "metric", x = 0, y = 6, width = 12, height = 6
        properties = {
          title  = "Stream consumer lag & errors"
          region = var.aws_region
          period = 60
          metrics = [
            ["AWS/Lambda", "IteratorAge", "FunctionName", aws_lambda_function.stream_consumer.function_name, { stat = "Maximum" }],
            [".", "Errors", ".", ".", { stat = "Sum" }]
          ]
        }
      },
      {
        type = "metric", x = 12, y = 6, width = 12, height = 6
        properties = {
          title  = "Investigations: verdicts, alerts, incomplete"
          region = var.aws_region
          stat   = "Sum", period = 300
          metrics = [
            ["RADAR", "VerdictFraud", "Backend", "bedrock"],
            [".", "VerdictLegit", ".", "."],
            [".", "VerdictReview", ".", "."],
            [".", "Alerts", ".", "."],
            [".", "Incomplete", ".", "."]
          ]
        }
      },
      {
        type = "metric", x = 0, y = 12, width = 12, height = 6
        properties = {
          title  = "Feature drift (max PSI, threshold ${var.psi_alert_threshold})"
          region = var.aws_region
          stat   = "Maximum", period = 3600
          metrics = [["RADAR", "MaxPSI", "ModelVersion", "ALL"]]
          annotations = { horizontal = [{ value = var.psi_alert_threshold, label = "retrain" }] }
        }
      },
      {
        type = "metric", x = 12, y = 12, width = 12, height = 6
        properties = {
          title  = "Agent tokens & latency"
          region = var.aws_region
          period = 300
          metrics = [
            ["RADAR", "AgentTokens", "Backend", "bedrock", { stat = "Average" }],
            [".", "AgentLatency", ".", ".", { stat = "p95", yAxis = "right" }]
          ]
        }
      }
    ]
  })
}
