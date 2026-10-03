"""Runtime configuration. Every value can be overridden with an environment variable.

Local development needs nothing set: defaults point at ./data and ./models.
In AWS, Terraform injects the resource names into the Lambda / ECS environment.
"""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="RADAR_", env_file=".env", extra="ignore")

    # ---- environment -------------------------------------------------------
    env: str = "local"  # local | dev | prod
    aws_region: str = "ap-south-1"
    log_level: str = "INFO"

    # ---- data & model locations -------------------------------------------
    data_dir: str = "data"
    models_dir: str = "models"
    # Either a local directory or an S3 prefix (s3://bucket/prefix). The scorer
    # reads <model_uri>/champion.json to find the live model version.
    model_uri: str = "models"
    model_refresh_seconds: int = 60

    # ---- streaming resources ----------------------------------------------
    kinesis_stream: str = "radar-transactions"
    firehose_feature_log: str = "radar-feature-log"
    user_state_table: str = "radar-user-state"
    decisions_table: str = "radar-decisions"
    cases_table: str = "radar-cases"
    investigation_queue_url: str = ""
    alerts_topic_arn: str = ""
    artefact_bucket: str = ""
    data_s3_prefix: str = ""  # s3://bucket/data - where the retrain task fetches labelled data

    # ---- scoring -------------------------------------------------------------
    scorer_url: str = "http://localhost:8000"
    scorer_api_key: str = ""  # sent as x-radar-key; the ALB listener rule requires it
    scorer_timeout_seconds: float = 2.0
    explain_only_flagged: bool = True

    # ---- drift & retraining ----------------------------------------------
    psi_alert_threshold: float = 0.20
    drift_lookback_hours: int = 24
    challenger_min_improvement: float = 0.01  # absolute gain in recall@1%FPR to promote

    # ---- agent -------------------------------------------------------------
    bedrock_region: str = "us-east-1"
    bedrock_model_id: str = "anthropic.claude-haiku-4-5-20251001-v1:0"
    agent_max_steps: int = 10
    agent_max_tokens: int = 1500
    agent_temperature: float = 0.0
    # USD per 1M tokens; update from the Bedrock pricing page for the model you use.
    price_input_per_mtok: float = 1.0
    price_output_per_mtok: float = 5.0


settings = Settings()
