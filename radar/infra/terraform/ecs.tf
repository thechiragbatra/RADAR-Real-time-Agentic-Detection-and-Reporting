resource "aws_ecs_cluster" "this" {
  name = local.name

  setting {
    name  = "containerInsights"
    value = "enabled"
  }
}

resource "aws_cloudwatch_log_group" "scorer" {
  name              = "/ecs/${local.name}-scorer"
  retention_in_days = 14
}

resource "aws_cloudwatch_log_group" "retrain" {
  name              = "/ecs/${local.name}-retrain"
  retention_in_days = 30
}

locals {
  scorer_image = "${aws_ecr_repository.scorer.repository_url}:${var.image_tag}"
  lambda_image = "${aws_ecr_repository.lambda.repository_url}:${var.image_tag}"
  model_uri    = "s3://${aws_s3_bucket.artefacts.bucket}/models"
}

# ---- Scoring service -------------------------------------------------------------------------
resource "aws_ecs_task_definition" "scorer" {
  family                   = "${local.name}-scorer"
  requires_compatibilities = ["FARGATE"]
  network_mode             = "awsvpc"
  cpu                      = var.scorer_cpu
  memory                   = var.scorer_memory
  execution_role_arn       = aws_iam_role.ecs_execution.arn
  task_role_arn            = aws_iam_role.scorer_task.arn

  runtime_platform {
    cpu_architecture        = "X86_64"
    operating_system_family = "LINUX"
  }

  container_definitions = jsonencode([{
    name      = "scorer"
    image     = local.scorer_image
    essential = true
    portMappings = [{
      containerPort = 8000
      protocol      = "tcp"
    }]
    environment = [
      { name = "RADAR_ENV", value = var.env },
      { name = "RADAR_AWS_REGION", value = var.aws_region },
      { name = "RADAR_MODEL_URI", value = local.model_uri },
      { name = "RADAR_MODEL_REFRESH_SECONDS", value = "60" },
      { name = "WEB_CONCURRENCY", value = "2" },
    ]
    healthCheck = {
      command     = ["CMD-SHELL", "python -c \"import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').status==200 else 1)\""]
      interval    = 30
      timeout     = 5
      retries     = 3
      startPeriod = 30
    }
    logConfiguration = {
      logDriver = "awslogs"
      options = {
        awslogs-group         = aws_cloudwatch_log_group.scorer.name
        awslogs-region        = var.aws_region
        awslogs-stream-prefix = "scorer"
      }
    }
  }])
}

resource "aws_ecs_service" "scorer" {
  name                               = "${local.name}-scorer"
  cluster                            = aws_ecs_cluster.this.id
  task_definition                    = aws_ecs_task_definition.scorer.arn
  desired_count                      = var.scorer_desired_count
  launch_type                        = "FARGATE"
  deployment_minimum_healthy_percent = 100
  deployment_maximum_percent         = 200
  health_check_grace_period_seconds  = 60

  deployment_circuit_breaker {
    enable   = true
    rollback = true
  }

  network_configuration {
    subnets          = aws_subnet.public[*].id
    security_groups  = [aws_security_group.scorer.id]
    assign_public_ip = true
  }

  load_balancer {
    target_group_arn = aws_lb_target_group.scorer.arn
    container_name   = "scorer"
    container_port   = 8000
  }

  depends_on = [aws_lb_listener.http]

  lifecycle {
    ignore_changes = [desired_count] # autoscaling owns it
  }
}

# ---- Autoscaling: request count per target + CPU ----------------------------------------
resource "aws_appautoscaling_target" "scorer" {
  max_capacity       = var.scorer_max_count
  min_capacity       = var.scorer_desired_count
  resource_id        = "service/${aws_ecs_cluster.this.name}/${aws_ecs_service.scorer.name}"
  scalable_dimension = "ecs:service:DesiredCount"
  service_namespace  = "ecs"
}

resource "aws_appautoscaling_policy" "scorer_requests" {
  name               = "${local.name}-scorer-requests"
  policy_type        = "TargetTrackingScaling"
  resource_id        = aws_appautoscaling_target.scorer.resource_id
  scalable_dimension = aws_appautoscaling_target.scorer.scalable_dimension
  service_namespace  = aws_appautoscaling_target.scorer.service_namespace

  target_tracking_scaling_policy_configuration {
    target_value       = 300 # requests/min per task; ~5 rps keeps p95 well under 50 ms
    scale_in_cooldown  = 120
    scale_out_cooldown = 60
    predefined_metric_specification {
      predefined_metric_type = "ALBRequestCountPerTarget"
      resource_label         = "${aws_lb.this.arn_suffix}/${aws_lb_target_group.scorer.arn_suffix}"
    }
  }
}

resource "aws_appautoscaling_policy" "scorer_cpu" {
  name               = "${local.name}-scorer-cpu"
  policy_type        = "TargetTrackingScaling"
  resource_id        = aws_appautoscaling_target.scorer.resource_id
  scalable_dimension = aws_appautoscaling_target.scorer.scalable_dimension
  service_namespace  = aws_appautoscaling_target.scorer.service_namespace

  target_tracking_scaling_policy_configuration {
    target_value = 60
    predefined_metric_specification {
      predefined_metric_type = "ECSServiceAverageCPUUtilization"
    }
  }
}

# ---- ALB ----------------------------------------------------------------------------------------------
resource "aws_lb" "this" {
  name               = "${local.name}-alb"
  load_balancer_type = "application"
  security_groups    = [aws_security_group.alb.id]
  subnets            = aws_subnet.public[*].id
}

resource "aws_lb_target_group" "scorer" {
  name        = "${local.name}-scorer"
  port        = 8000
  protocol    = "HTTP"
  vpc_id      = aws_vpc.this.id
  target_type = "ip"

  deregistration_delay = 15

  health_check {
    path                = "/health"
    interval            = 15
    timeout             = 5
    healthy_threshold   = 2
    unhealthy_threshold = 3
    matcher             = "200"
  }
}

# Requests without the shared secret header get a 403 from the ALB itself.
resource "aws_lb_listener" "http" {
  load_balancer_arn = aws_lb.this.arn
  port              = 80
  protocol          = "HTTP"

  default_action {
    type = "fixed-response"
    fixed_response {
      content_type = "application/json"
      message_body = "{\"detail\":\"forbidden\"}"
      status_code  = "403"
    }
  }
}

resource "aws_lb_listener_rule" "keyed" {
  listener_arn = aws_lb_listener.http.arn
  priority     = 10

  action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.scorer.arn
  }

  condition {
    http_header {
      http_header_name = "x-radar-key"
      values           = [var.scorer_api_key]
    }
  }
}

# ---- Retraining task (run on demand by EventBridge) -------------------------------------
resource "aws_ecs_task_definition" "retrain" {
  family                   = "${local.name}-retrain"
  requires_compatibilities = ["FARGATE"]
  network_mode             = "awsvpc"
  cpu                      = 2048
  memory                   = 8192
  execution_role_arn       = aws_iam_role.ecs_execution.arn
  task_role_arn            = aws_iam_role.retrain_task.arn

  container_definitions = jsonencode([{
    name       = "retrain"
    image      = local.lambda_image
    essential  = true
    entryPoint = ["/var/lang/bin/python3", "-m", "radar.model.retrain"]
    command    = ["--data", "/tmp/data", "--registry", local.model_uri, "--workdir", "/tmp/models", "--promote-if-better", "--reason", "drift"]
    environment = [
      { name = "RADAR_ENV", value = var.env },
      { name = "RADAR_AWS_REGION", value = var.aws_region },
      { name = "RADAR_MODEL_URI", value = local.model_uri },
      { name = "RADAR_ARTEFACT_BUCKET", value = aws_s3_bucket.artefacts.bucket },
      { name = "RADAR_DATA_S3_PREFIX", value = "s3://${aws_s3_bucket.artefacts.bucket}/data" },
    ]
    logConfiguration = {
      logDriver = "awslogs"
      options = {
        awslogs-group         = aws_cloudwatch_log_group.retrain.name
        awslogs-region        = var.aws_region
        awslogs-stream-prefix = "retrain"
      }
    }
  }])
}
