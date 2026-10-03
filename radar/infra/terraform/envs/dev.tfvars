project              = "radar"
env                  = "dev"
aws_region           = "ap-south-1"
bedrock_region       = "us-east-1"
bedrock_model_id     = "anthropic.claude-haiku-4-5-20251001-v1:0"
scorer_desired_count = 1
scorer_max_count     = 4
kinesis_shards       = 1
psi_alert_threshold  = 0.2
# scorer_api_key and alert_email: pass with -var or TF_VAR_*, never commit them.
