# Runbook: deploy, operate, tear down

## Prerequisites

* AWS account with Bedrock model access enabled (console -> Bedrock -> Model access) in
  `bedrock_region` (default `us-east-1`; Mumbai users typically use an `apac.` inference profile).
* Terraform >= 1.6, Docker, AWS CLI v2, Python 3.11+.
* For CI/CD: a GitHub OIDC role (`AWS_DEPLOY_ROLE_ARN`), `SCORER_API_KEY` secret, `AWS_REGION` var.

## 1. Build the model locally

```
make setup simulate train         # ~1.5 min; prints the model report
make test                         # 42 tests, ~1 min
```

## 2. Provision

```
export TF_VAR_scorer_api_key=$(openssl rand -hex 24)
export TF_VAR_alert_email=you@example.com       # optional; confirm the SNS subscription email
make tf-init
make tf-apply                                   # ~6 min; creates ~60 resources
```

The first apply fails on the ECS service / Lambdas because the ECR repositories are empty.
Push images, then apply again (the deploy workflow does this ordering automatically):

```
cd infra/terraform && terraform apply -target=aws_ecr_repository.scorer -target=aws_ecr_repository.lambda -var-file=envs/dev.tfvars
ECR_SCORER=$(terraform output -raw ecr_scorer); ECR_LAMBDA=$(terraform output -raw ecr_lambda); cd -
aws ecr get-login-password | docker login --username AWS --password-stdin ${ECR_SCORER%%/*}
docker build -f docker/Dockerfile.scorer -t $ECR_SCORER:latest . && docker push $ECR_SCORER:latest
docker build -f docker/Dockerfile.lambda -t $ECR_LAMBDA:latest . && docker push $ECR_LAMBDA:latest
make tf-apply
```

## 3. Bootstrap data and models

```
cd infra/terraform && BUCKET=$(terraform output -raw artefact_bucket); TABLE=$(terraform output -raw state_table); cd -
make bootstrap-aws BUCKET=$BUCKET TABLE=$TABLE
```

Uploads every local model version, promotes the local champion, loads 15k user profiles and
1.5k merchants into the state table, and copies the dataset to `s3://$BUCKET/data/` for retraining.

## 4. Run traffic

```
radar-produce --stream radar-dev-transactions --rate 200 --offset 520000 --limit 50000
```

Start at an offset past the training window so the stream contains unseen transactions. Watch:

* CloudWatch dashboard (`terraform output dashboard_url`): throughput, flags, latency, lag
* `aws logs tail /aws/lambda/radar-dev-investigate --follow` for agent traces
* DynamoDB `radar-dev-cases` for verdicts and reports; SNS email for high-confidence fraud

## 5. Evaluate the agent on AWS

```
AWS_PROFILE=... RADAR_BEDROCK_MODEL_ID=anthropic.claude-haiku-4-5-20251001-v1:0 make eval-bedrock
make eval-compare
```

Runs the 100 labelled cases through Bedrock from your laptop (local data source, real LLM) and
writes `evals/results/comparison.md`. Budget ~100 x 6k tokens = ~600k tokens.

## 6. Force a drift event (demo)

```
aws lambda invoke --function-name radar-dev-drift-monitor /dev/stdout
aws events put-events --entries '[{"Source":"radar.drift","DetailType":"DriftDetected","Detail":"{\"max_psi\":0.31,\"max_feature\":\"amount\"}"}]'
aws ecs list-tasks --cluster radar-dev --family radar-dev-retrain
```

The retrain task logs to `/ecs/radar-dev-retrain` and writes
`s3://$BUCKET/models/decisions/<version>.json` with the promote/keep decision.

## Operating notes

* **Scorer model refresh**: 60 s poll on `champion.json` ETag. Promotion = one S3 write.
* **Rollback**: `ModelRegistry("s3://.../models").promote("<old version>", "rollback")` from a
  Python shell, or edit `champion.json`. Takes effect within a minute.
* **Consumer lag**: `IteratorAge` alarm at 60 s. Scale by adding shards (`kinesis_shards`); the
  Lambda scales with shards automatically.
* **Investigation backlog**: FIFO queue depth + reserved concurrency 5. Raise concurrency or Bedrock
  quotas if the queue grows; poison messages go to the DLQ after 3 attempts and raise an alarm.
* **Kill switch**: set the investigate Lambda's `RADAR_BEDROCK_MODEL_ID=heuristic` to fall back to
  the rule-based investigator without a deploy.

## Cost (ap-south-1, demo load, approximate)

| item | monthly |
|---|---|
| Fargate scorer 0.5 vCPU / 1 GB, 1 task | ~$15 |
| ALB | ~$18 + data |
| Kinesis 1 shard | ~$11 |
| DynamoDB on-demand, S3, Firehose, Lambda, SQS, SNS, CloudWatch at demo volume | < $5 |
| Bedrock (Haiku-class model, ~6k tokens/case, 100 cases/day) | ~$10 |
| **total while running** | **~$60 / ~Rs 5,000** |

Stop the ECS service and Kinesis between demos and the bill drops to a few dollars. Free-tier
accounts cover most of DynamoDB/Lambda/S3 for 12 months.

## Tear down

```
make tf-destroy        # buckets have force_destroy = true; ECR repos force_delete = true
```
