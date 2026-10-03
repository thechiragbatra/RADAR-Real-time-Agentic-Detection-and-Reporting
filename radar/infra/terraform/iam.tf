data "aws_iam_policy_document" "assume" {
  for_each = {
    ecs_tasks = "ecs-tasks.amazonaws.com"
    lambda    = "lambda.amazonaws.com"
    firehose  = "firehose.amazonaws.com"
    events    = "events.amazonaws.com"
  }
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = [each.value]
    }
  }
}

# ---- ECS task execution (pull image, write logs) ---------------------------------------------
resource "aws_iam_role" "ecs_execution" {
  name               = "${local.name}-ecs-execution"
  assume_role_policy = data.aws_iam_policy_document.assume["ecs_tasks"].json
}

resource "aws_iam_role_policy_attachment" "ecs_execution" {
  role       = aws_iam_role.ecs_execution.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy"
}

# ---- Scorer task role: read the model registry -------------------------------------------
resource "aws_iam_role" "scorer_task" {
  name               = "${local.name}-scorer-task"
  assume_role_policy = data.aws_iam_policy_document.assume["ecs_tasks"].json
}

resource "aws_iam_role_policy" "scorer_task" {
  role = aws_iam_role.scorer_task.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect   = "Allow"
      Action   = ["s3:GetObject", "s3:ListBucket"]
      Resource = [aws_s3_bucket.artefacts.arn, "${aws_s3_bucket.artefacts.arn}/models/*"]
    }]
  })
}

# ---- Retrain task role: read data, read/write the registry -------------------------------
resource "aws_iam_role" "retrain_task" {
  name               = "${local.name}-retrain-task"
  assume_role_policy = data.aws_iam_policy_document.assume["ecs_tasks"].json
}

resource "aws_iam_role_policy" "retrain_task" {
  role = aws_iam_role.retrain_task.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect   = "Allow"
        Action   = ["s3:GetObject", "s3:PutObject", "s3:ListBucket"]
        Resource = [aws_s3_bucket.artefacts.arn, "${aws_s3_bucket.artefacts.arn}/*"]
      },
      {
        Effect   = "Allow"
        Action   = ["cloudwatch:PutMetricData"]
        Resource = "*"
      }
    ]
  })
}

# ---- Lambda role (shared by the three functions; least privilege per resource) ------
resource "aws_iam_role" "lambda" {
  name               = "${local.name}-lambda"
  assume_role_policy = data.aws_iam_policy_document.assume["lambda"].json
}

resource "aws_iam_role_policy_attachment" "lambda_basic" {
  role       = aws_iam_role.lambda.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
}

resource "aws_iam_role_policy" "lambda" {
  role = aws_iam_role.lambda.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = ["kinesis:GetRecords", "kinesis:GetShardIterator", "kinesis:DescribeStream",
        "kinesis:DescribeStreamSummary", "kinesis:ListShards", "kinesis:ListStreams", "kinesis:SubscribeToShard"]
        Resource = aws_kinesis_stream.transactions.arn
      },
      {
        Effect   = "Allow"
        Action   = ["dynamodb:GetItem", "dynamodb:PutItem", "dynamodb:UpdateItem", "dynamodb:Query", "dynamodb:BatchGetItem"]
        Resource = [aws_dynamodb_table.state.arn, aws_dynamodb_table.decisions.arn, "${aws_dynamodb_table.decisions.arn}/index/*", aws_dynamodb_table.cases.arn]
      },
      {
        Effect   = "Allow"
        Action   = ["sqs:SendMessage", "sqs:ReceiveMessage", "sqs:DeleteMessage", "sqs:GetQueueAttributes"]
        Resource = [aws_sqs_queue.investigations.arn, aws_sqs_queue.consumer_failures.arn]
      },
      {
        Effect   = "Allow"
        Action   = ["firehose:PutRecord", "firehose:PutRecordBatch"]
        Resource = aws_kinesis_firehose_delivery_stream.feature_log.arn
      },
      {
        Effect   = "Allow"
        Action   = ["sns:Publish"]
        Resource = aws_sns_topic.alerts.arn
      },
      {
        Effect   = "Allow"
        Action   = ["s3:GetObject", "s3:PutObject", "s3:ListBucket"]
        Resource = [aws_s3_bucket.artefacts.arn, "${aws_s3_bucket.artefacts.arn}/*"]
      },
      {
        Effect   = "Allow"
        Action   = ["cloudwatch:PutMetricData", "events:PutEvents"]
        Resource = "*"
      },
      {
        Effect   = "Allow"
        Action   = ["bedrock:InvokeModel", "bedrock:InvokeModelWithResponseStream", "bedrock:Converse", "bedrock:ConverseStream"]
        Resource = "*"
      }
    ]
  })
}

# ---- Firehose role --------------------------------------------------------------------------------
resource "aws_iam_role" "firehose" {
  name               = "${local.name}-firehose"
  assume_role_policy = data.aws_iam_policy_document.assume["firehose"].json
}

resource "aws_iam_role_policy" "firehose" {
  role = aws_iam_role.firehose.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect   = "Allow"
      Action   = ["s3:AbortMultipartUpload", "s3:GetBucketLocation", "s3:GetObject", "s3:ListBucket", "s3:ListBucketMultipartUploads", "s3:PutObject"]
      Resource = [aws_s3_bucket.artefacts.arn, "${aws_s3_bucket.artefacts.arn}/*"]
    }]
  })
}

# ---- EventBridge -> ECS RunTask --------------------------------------------------------------
resource "aws_iam_role" "events" {
  name               = "${local.name}-events"
  assume_role_policy = data.aws_iam_policy_document.assume["events"].json
}

resource "aws_iam_role_policy" "events" {
  role = aws_iam_role.events.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect   = "Allow"
        Action   = ["ecs:RunTask"]
        Resource = aws_ecs_task_definition.retrain.arn
        Condition = {
          ArnLike = { "ecs:cluster" = aws_ecs_cluster.this.arn }
        }
      },
      {
        Effect   = "Allow"
        Action   = ["iam:PassRole"]
        Resource = [aws_iam_role.ecs_execution.arn, aws_iam_role.retrain_task.arn]
      }
    ]
  })
}
