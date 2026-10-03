# ---- S3: model registry + feature log --------------------------------------------------
resource "aws_s3_bucket" "artefacts" {
  bucket        = "${local.name}-artefacts-${local.account_id}"
  force_destroy = true
}

resource "aws_s3_bucket_versioning" "artefacts" {
  bucket = aws_s3_bucket.artefacts.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_public_access_block" "artefacts" {
  bucket                  = aws_s3_bucket.artefacts.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_lifecycle_configuration" "artefacts" {
  bucket = aws_s3_bucket.artefacts.id

  rule {
    id     = "expire-feature-log"
    status = "Enabled"
    filter {
      prefix = "feature-log/"
    }
    expiration {
      days = 30
    }
  }

  rule {
    id     = "expire-old-model-versions"
    status = "Enabled"
    filter {
      prefix = "models/"
    }
    noncurrent_version_expiration {
      noncurrent_days = 90
    }
  }
}

# ---- DynamoDB ------------------------------------------------------------------------------------
# One table for rolling user/device state and static reference data (pk = "USER#", "DEVICE#",
# "PROFILE#", "MERCHANT#"). Single-digit-millisecond reads on the hot path.
resource "aws_dynamodb_table" "state" {
  name         = "${local.name}-user-state"
  billing_mode = "PAY_PER_REQUEST"
  hash_key     = "pk"

  attribute {
    name = "pk"
    type = "S"
  }

  point_in_time_recovery {
    enabled = true
  }
}

resource "aws_dynamodb_table" "decisions" {
  name         = "${local.name}-decisions"
  billing_mode = "PAY_PER_REQUEST"
  hash_key     = "txn_id"

  attribute {
    name = "txn_id"
    type = "S"
  }
  attribute {
    name = "user_id"
    type = "S"
  }
  attribute {
    name = "ts"
    type = "N"
  }

  # the agent's get_user_transactions tool: a user's recent decisions in time order
  global_secondary_index {
    name            = "by_user"
    hash_key        = "user_id"
    range_key       = "ts"
    projection_type = "ALL"
  }

  ttl {
    attribute_name = "expires_at"
    enabled        = true
  }
}

resource "aws_dynamodb_table" "cases" {
  name         = "${local.name}-cases"
  billing_mode = "PAY_PER_REQUEST"
  hash_key     = "case_id"

  attribute {
    name = "case_id"
    type = "S"
  }
  attribute {
    name = "txn_id"
    type = "S"
  }

  global_secondary_index {
    name            = "by_txn"
    hash_key        = "txn_id"
    projection_type = "ALL"
  }
}

# ---- Kinesis + Firehose -------------------------------------------------------------------
resource "aws_kinesis_stream" "transactions" {
  name             = "${local.name}-transactions"
  shard_count      = var.kinesis_shards
  retention_period = 24

  stream_mode_details {
    stream_mode = "PROVISIONED"
  }
}

resource "aws_kinesis_firehose_delivery_stream" "feature_log" {
  name        = "${local.name}-feature-log"
  destination = "extended_s3"

  extended_s3_configuration {
    role_arn            = aws_iam_role.firehose.arn
    bucket_arn          = aws_s3_bucket.artefacts.arn
    prefix              = "feature-log/!{timestamp:yyyy/MM/dd/HH}/"
    error_output_prefix = "feature-log-errors/!{firehose:error-output-type}/!{timestamp:yyyy/MM/dd}/"
    buffering_size      = 1
    buffering_interval  = 60
    compression_format  = "UNCOMPRESSED"
  }
}

# ---- SQS + SNS ------------------------------------------------------------------------------------
resource "aws_sqs_queue" "investigations_dlq" {
  name       = "${local.name}-investigations-dlq.fifo"
  fifo_queue = true
}

resource "aws_sqs_queue" "investigations" {
  name                        = "${local.name}-investigations.fifo"
  fifo_queue                  = true
  content_based_deduplication = false
  visibility_timeout_seconds  = 360 # > lambda timeout
  message_retention_seconds   = 86400 * 4

  redrive_policy = jsonencode({
    deadLetterTargetArn = aws_sqs_queue.investigations_dlq.arn
    maxReceiveCount     = 3
  })
}

resource "aws_sqs_queue" "consumer_failures" {
  name = "${local.name}-consumer-failures"
}

resource "aws_sns_topic" "alerts" {
  name = "${local.name}-alerts"
}

resource "aws_sns_topic_subscription" "email" {
  count     = var.alert_email == "" ? 0 : 1
  topic_arn = aws_sns_topic.alerts.arn
  protocol  = "email"
  endpoint  = var.alert_email
}

# ---- ECR ----------------------------------------------------------------------------------------------
resource "aws_ecr_repository" "scorer" {
  name                 = "${local.name}-scorer"
  image_tag_mutability = "MUTABLE"
  force_delete         = true
  image_scanning_configuration {
    scan_on_push = true
  }
}

resource "aws_ecr_repository" "lambda" {
  name                 = "${local.name}-lambda"
  image_tag_mutability = "MUTABLE"
  force_delete         = true
  image_scanning_configuration {
    scan_on_push = true
  }
}
