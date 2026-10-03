terraform {
  required_version = ">= 1.6"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.60"
    }
  }

  # Uncomment for shared state (create the bucket + table once by hand or with a bootstrap stack).
  # backend "s3" {
  #   bucket         = "radar-tfstate-<account-id>"
  #   key            = "radar/dev/terraform.tfstate"
  #   region         = "ap-south-1"
  #   dynamodb_table = "radar-tfstate-lock"
  #   encrypt        = true
  # }
}

provider "aws" {
  region = var.aws_region

  default_tags {
    tags = {
      Project     = var.project
      Environment = var.env
      ManagedBy   = "terraform"
    }
  }
}

data "aws_caller_identity" "current" {}
data "aws_region" "current" {}

locals {
  name       = "${var.project}-${var.env}"
  account_id = data.aws_caller_identity.current.account_id
}
