# Design decisions

Short ADR-style notes. Each one is a trade-off an interviewer can push on.

## 1. Time-based split with burn-in, not a random split

Random splits leak: a user's later transactions shape the feature store that scores their earlier
ones, and the same fraud episode lands in both train and test. The split is by time (train / 10-day
validation / 14-day test) and the first 14 days are replayed for state only. The operating threshold
is chosen on validation and *reported* on test, including the FPR it actually achieved (0.99% for a
1% target in the shipped model).

## 2. `scale_pos_weight` instead of SMOTE

SMOTE interpolates between fraud rows in feature space and produces transactions that never happen
(half-way between a card-testing burst and a geo-jump). Reweighting changes the loss, not the data.
We use `sqrt(neg/pos)` rather than the full ratio: the full ratio pushes scores up across the board
and makes the probability output useless for ranking alerts.

## 3. Recall at fixed FPR is the headline metric

Accuracy is 99.3% for a model that flags nothing. A fraud team cares about how much fraud is caught
when only 1 in 100 legitimate customers is bothered, and how many alerts that generates per analyst
per day. ROC-AUC and PR-AUC are reported for comparability, but the deployed threshold is a business
choice (1% FPR -> ~100 alerts/day at this volume).

## 4. Same feature code for training and serving

The alternative - a SQL/Spark feature pipeline offline and a hand-written one online - is where most
production fraud systems get train/serve skew. Here one Python function is called by both the
replay and the Lambda, and `tests/test_features.py::test_replay_has_no_leakage_and_matches_online_path`
asserts they agree row for row. The cost is a pure-Python replay (~50 us/row, 30 s for 600k rows),
which is fine at this scale and parallelisable by user if it is not.

## 5. DynamoDB as the online feature store, no locking

Kinesis partitions by `user_id`, Lambda runs one invocation per shard, so a user's records are
processed sequentially and the read-modify-write on `UserState` needs no lock. Device state can be
touched from two shards concurrently (two users sharing a device); we accept the rare lost update
because `device_user_count` is a soft signal. Redis would be faster but needs a VPC, a NAT or
endpoints, and persistence configuration; DynamoDB on-demand is single-digit ms and ~free at demo
volume.

## 6. Scoring service on Fargate vs model inside the Lambda

The model is small enough to load in the Lambda (`RADAR_SCORER_URL=local` does exactly that), and
that variant removes the ALB hop. We kept the service because (a) synchronous callers - a checkout
flow asking "should I step up auth?" - need an endpoint, (b) it scales on request rate independently
of the stream consumer, and (c) hot-swapping a model in one place is simpler than invalidating N
warm Lambdas. The hop costs ~5-15 ms inside a region.

## 7. Public ALB gated by a header, no NAT gateway

A NAT gateway is ~$32/month before data; interface endpoints for Kinesis, SQS, Firehose, Bedrock
and CloudWatch add ~$7/month each. For a demo that is the entire budget. Tasks run in public
subnets with a security group that only admits the ALB, and the ALB listener returns 403 unless
`x-radar-key` matches. Production hardening: private subnets, internal ALB, Lambdas in the VPC,
gateway endpoints (free) for S3/DynamoDB and interface endpoints for the rest, HTTPS with ACM.

## 8. PSI on features *and* on the score

Labels arrive late (chargebacks take weeks), so monitoring has to be label-free. Feature PSI catches
input shifts (a new merchant category, a payments outage changing velocity). Score PSI catches the
case where every feature looks stable but the model's output distribution moves - usually a sign of
a broken upstream field. Both use bins frozen at training time, so the comparison is always against
what the model actually learned on.

## 9. Champion/challenger with a one-sided promotion test

The challenger must gain at least +1 point of recall at 1% FPR *and* not lose PR-AUC. Silent
regressions cost more than missed gains, and retraining on a window that happens to contain an
easy fraud wave would otherwise promote a worse model. Every decision - promoted or not - is written
to the registry with both models' metrics, which is the audit trail a model-risk team asks for.

## 10. The agent finishes only through a validated tool call

Free-text verdicts need parsing, and parsing LLM output is where "it worked in the demo" dies. The
model must call `submit_verdict` with a payload that validates against `InvestigationResult`
(enum verdict, bounded confidence, 1-8 evidence strings, action enum, summary length). Invalid
payloads go back as tool errors with the validation message; after two retries the case is routed
to a human. The Lambda, not the model, decides what happens to the account.

## 11. A heuristic investigator as the baseline

An LLM agent that is 75% accurate sounds good until you learn six if-statements get 70%. The
heuristic backend drives the same tools with a fixed plan and rule-based scoring, so the comparison
is like for like: same cases, same tools, same output schema. It also doubles as a zero-cost
fallback if Bedrock is unavailable (`RADAR_BEDROCK_MODEL_ID=heuristic`).

## 12. Bedrock Converse directly, no agent framework

LangChain/LangGraph would add a dependency layer between us and the guardrails we need to reason
about (allow-list, retries, budgets). The Converse API already provides tool use with JSON schemas,
so the loop is ~120 lines we fully own and can unit-test with a scripted backend.

## 13. Synthetic data, and why the numbers are still meaningful

Real transaction data cannot be published. The simulator is built so that the model cannot win by
cheating: victims are drawn in proportion to activity (otherwise "low activity" leaks the label),
fraud multipliers are modest (1.2-4x, not 10x), legitimate life events (trips, phone upgrades,
sale-day sprees, bill-pay bursts, night owls, shared family phones) are the false-positive drivers
you see in production, and the hardest pattern (`low_and_slow`) stays inside the user's normal
spend. The absolute recall is a property of the simulator; the *gaps* - rules vs LR vs XGBoost,
per-pattern recall, which legit behaviours get flagged - are the transferable findings.
