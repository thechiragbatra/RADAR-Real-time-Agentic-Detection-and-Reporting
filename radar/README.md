# RADAR — Real-time Agentic Detection and Reporting

*Real-time fraud detection with an agentic investigation layer, on AWS.*

A production-shaped fraud detection system: transactions are scored on a stream in under a
millisecond server-side, the model watches its own feature drift and retrains champion/challenger
style, and every flagged transaction is investigated by an LLM agent that gathers evidence with
tools and commits a schema-validated verdict. Infrastructure is Terraform, delivery is GitHub
Actions, and both the model and the agent are evaluated against baselines on held-out data.

```
Kinesis ──► Lambda (features from DynamoDB state) ──► FastAPI/XGBoost on Fargate ──► decisions, feature log
                                                              │
                                   flagged ──► SQS ──► Lambda + Bedrock agent (tools) ──► verdict, report, alert
                                                              │
      S3 feature log ──► hourly PSI drift monitor ──► EventBridge ──► retrain task ──► promote if better ──► hot-swap
```

Full diagram and walkthrough: [docs/architecture.md](docs/architecture.md). Full model report: [docs/model-report-v1.md](docs/model-report-v1.md).

## Results

Dataset: 606k synthetic transactions over 90 days for 15k users, 0.7% fraud across six injected
patterns, plus the legitimate behaviours that make fraud detection hard (trips, phone upgrades,
sale-day sprees, bill-pay bursts, night owls, shared family phones). Time-based split; threshold
chosen on validation at a 1% false-positive target; everything below is on the untouched 14-day
test window (81k transactions, 608 fraud).

| model | ROC-AUC | PR-AUC | recall @ 0.5% FPR | recall @ 1% FPR | recall @ 2% FPR | deployed operating point | precision | alerts/day |
|---|---|---|---|---|---|---|---|---|
| hand-written rules | 0.814 | 0.301 | 66.8% | 66.8% | 66.8% | 66.8% @ 5.89% FPR | 7.9% | 369 |
| logistic regression | 0.956 | 0.676 | 69.4% | 73.7% | 77.0% | 73.7% @ 0.96% FPR | 36.7% | 87 |
| **XGBoost** | **0.982** | **0.882** | **88.0%** | **89.6%** | **91.6%** | **89.6% @ 0.99% FPR** | **40.6%** | **96** |

Recall by fraud pattern at the deployed threshold: account takeover 100%, card testing 100%,
geo jump 100%, low-and-slow 97.9%, merchant collusion 61.8%, velocity burst 61.4%. The two weak
patterns are the ones with no device or location signal - exactly where the investigation layer
earns its place.

False positives by legitimate behaviour (test window): travellers are flagged 6x more often than
regular transactions (3.2% vs 0.5%), sale-day sprees 2.6x, phone upgrades 2.3x.

Scoring service: 0.4 ms median server-side, ~1.5 ms when SHAP reasons are computed for a flagged
transaction; 717 req/s with p95 63 ms end to end on two uvicorn workers in the load test
([loadtest/README.md](loadtest/README.md)).

Investigation agent, heuristic baseline on 100 labelled cases (50 fraud, 50 false positives):
decides 57% of cases, 82.5% accurate when it does, 7 false blocks, 3 missed frauds, 7 tool calls
per case, 40 ms. The LLM column fills in when you run `make eval-bedrock` with your own account —
[evals/README.md](evals/README.md).

> These numbers are properties of a simulator built to be adversarial (see
> [docs/decisions.md](docs/decisions.md#13-synthetic-data-and-why-the-numbers-are-still-meaningful)).
> What transfers to real data is the methodology, the gaps between models, and the failure analysis.

## What is in the box

| area | what | where |
|---|---|---|
| Simulator | users, merchants, devices; six fraud patterns; hard-negative legit behaviours; deterministic | `src/radar/simulator/` |
| Features | 45 causal features from bounded per-user/device rolling state; one code path for training and serving; in-memory and DynamoDB stores | `src/radar/features/` |
| Model | XGBoost with time-based split, burn-in, imbalance handling, rules + LR baselines, recall@FPR, per-pattern recall, SHAP, PSI reference, case bank; registry with champion pointer; champion/challenger retrain | `src/radar/model/` |
| Scoring service | FastAPI; hot-swaps the champion from S3; SHAP reasons only for flagged; Prometheus metrics | `src/radar/scoring/` |
| Stream path | Kinesis consumer Lambda with partial-batch failures, Firehose feature log, FIFO queue for flagged | `src/radar/lambdas/` |
| Agent | Bedrock Converse tool-use loop with allow-list, argument validation, step/token budgets, structured verdict; heuristic baseline backend; analyst report | `src/radar/agent/` |
| Drift | PSI per feature and on the score vs frozen training bins; CloudWatch metrics; EventBridge trigger | `src/radar/model/drift.py`, `lambdas/drift_monitor.py` |
| Evals | stratified 100-case set; coverage / decided accuracy / false blocks / missed fraud / cost; backend comparison | `src/radar/evals/`, `evals/` |
| Infra | VPC, Kinesis, Firehose, DynamoDB x3, SQS FIFO + DLQ, SNS, S3, ECR, ECS Fargate + ALB + autoscaling, 3 Lambdas, EventBridge, CloudWatch alarms + dashboard, IAM | `infra/terraform/` |
| Delivery | CI (ruff, pytest, terraform validate, docker build); OIDC deploy (ECR push, terraform apply, smoke test) | `.github/workflows/` |
| Tests | 42 tests: simulator invariants, feature math, no train/serve skew, model artefacts, API contract and hot-swap, consumer end-to-end on moto, agent guardrails, drift, eval harness | `tests/` |

## Run it locally (no AWS needed)

```
pip install -e ".[dev]"
make simulate        # 606k transactions, ~20 s
make train           # trains, evaluates, prints the report, promotes v1 (~1 min)
make test            # 42 tests
make demo            # streams 3000 transactions through the real scoring path and investigates the flags
make serve           # scoring API on :8000  ->  curl localhost:8000/health
radar-investigate --random-flagged 3        # full investigation report for three flagged transactions
```

`make demo` output looks like this:

```
  FLAG T000604166 score=0.847 Rs    3,087 utilities    [device_user_age_days=4.91, device_user_count=7, hour_local=17.2]  -> FRAUD/low_and_slow
  FLAG T000606206 score=0.137 Rs    2,208 ecommerce    [hour_local=23, seconds_since_last=664, device_user_age_days=89.2]  -> legit/sale_spree

3,000 scored in 3.2s | flagged 10 (0.33%) | end-to-end latency p50 0.5 ms p95 0.8 ms (feature store + score + SHAP for flagged)
flagged: 3 fraud, 7 false positives | fraud in window: 3

  [ok ] T000604166 verdict=fraud        conf=0.95 action=block_card_and_contact_customer  truth=FRAUD/low_and_slow
        device has 25 confirmed fraud transactions
  [BAD] T000603411 verdict=fraud        conf=0.81 action=block_card_and_contact_customer  truth=legit/big_purchase
        amount Rs16,891 is 18.0x the account's 2-week median Rs939
  [?  ] T000605224 verdict=needs_review conf=0.50 action=step_up_authentication           truth=legit/regular
        71% of the most similar resolved cases were fraud
```

The `[BAD]` line is the point of the evaluation harness: the rule-based investigator blocks a
customer for a Rs 16,891 travel booking from a two-day-old device. The account's history shows
an Rs 11,779 electronics purchase on the old phone two days earlier, followed by small food
orders from the new one - a phone upgrade. An LLM agent that reads the history is expected to
send this to review rather than block it; the eval harness measures whether it does.

## Deploy to AWS

[docs/runbook.md](docs/runbook.md) — Terraform apply, image push, bootstrap, traffic replay,
forcing a drift event, cost (~$60/month while running, near zero when stopped), tear-down.

## Deploy the scorer on Render

The root-level [`render.yaml`](../render.yaml) defines a Docker web service in Singapore on
Render's Starter plan. In Render,
create a Blueprint from this repository. Render generates a secret `RADAR_SCORER_API_KEY`
automatically. The image builds the synthetic training data and model during deployment, so
no ignored local model files or persistent disk are needed. This makes image builds longer; each
deployment gets its own fresh model, and runtime model promotion is not persisted.

The `/health` route is public for Render's health checks. Send the configured secret in the
`x-radar-key` header for every other route, for example:

```
curl -H "x-radar-key: $RADAR_SCORER_API_KEY" https://<your-render-service>.onrender.com/model
```

Render's Starter plan is a paid service. Keep the API key in Render's environment settings; do not
commit it or put it in a client-side application.

## Repository layout

```
src/radar/
  simulator/   world.py, fraud_patterns.py, generate.py, producer.py
  features/    state.py, store.py, reference.py, engineering.py
  model/       train.py, evaluate.py, drift.py, registry.py, retrain.py
  scoring/     scorer.py, api.py
  agent/       investigator.py, tools.py, llm.py, datasource.py, prompts.py, report.py, cli.py
  lambdas/     stream_consumer.py, investigate.py, drift_monitor.py, common.py
  evals/       harness.py
infra/terraform/   providers, variables, vpc, storage, iam, ecs, lambda, monitoring, outputs
docker/            Dockerfile.scorer, Dockerfile.lambda
.github/workflows/ ci.yml, deploy.yml
tests/             42 tests
docs/              architecture.md, decisions.md, runbook.md, interview-notes.md
evals/             cases/, results/, README.md
loadtest/          locustfile.py, results/
scripts/           local_stream_demo.py, bootstrap_aws.py
```

## Roadmap

* Graph features over the user–device–merchant graph for ring detection (today approximated by
  device sharing counts)
* Merchant fraud rate as a time-lagged rolling feature instead of a static category prior
* Label feed from the cases table + chargebacks into retraining; analyst feedback on agent verdicts
  as a second evaluation signal
* Production hardening: private subnets + internal ALB + VPC endpoints, HTTPS, per-tenant API keys
* Streaming feature computation in Flink (Kinesis Data Analytics) if volume outgrows one Lambda per shard

## Licence

MIT
