variable "project" {
  type    = string
  default = "radar"
}

variable "env" {
  type    = string
  default = "dev"
}

variable "aws_region" {
  type    = string
  default = "ap-south-1"
}

variable "bedrock_region" {
  description = "Region used for Bedrock calls (model availability differs by region)."
  type        = string
  default     = "us-east-1"
}

variable "bedrock_model_id" {
  description = "Bedrock model id or inference profile id for the investigation agent."
  type        = string
  default     = "anthropic.claude-haiku-4-5-20251001-v1:0"
}

variable "image_tag" {
  description = "Container image tag for the scorer and lambda images (set by CI to the git sha)."
  type        = string
  default     = "latest"
}

variable "scorer_cpu" {
  type    = number
  default = 512
}

variable "scorer_memory" {
  type    = number
  default = 1024
}

variable "scorer_desired_count" {
  type    = number
  default = 1
}

variable "scorer_max_count" {
  type    = number
  default = 6
}

variable "scorer_api_key" {
  description = "Shared secret the ALB requires in the x-radar-key header. Rotate by changing it."
  type        = string
  sensitive   = true
}

variable "kinesis_shards" {
  type    = number
  default = 1
}

variable "alert_email" {
  description = "Optional email for SNS fraud alerts and CloudWatch alarms."
  type        = string
  default     = ""
}

variable "psi_alert_threshold" {
  type    = number
  default = 0.2
}

variable "vpc_cidr" {
  type    = string
  default = "10.42.0.0/16"
}
