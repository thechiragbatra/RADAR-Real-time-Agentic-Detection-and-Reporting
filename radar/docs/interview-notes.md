# Talking about RADAR in an interview

RADAR = **R**eal-time **A**gentic **D**etection **a**nd **R**eporting. Say the expansion once, early; it is the whole architecture in five words.

## The 30-second version

> Fraud models in production fail quietly: the data drifts, nobody notices until the losses show
> up, and every alert still needs a human to work it. I built RADAR - Real-time Agentic Detection
> and Reporting - a fraud detection system on AWS. Transactions come in on a Kinesis stream, a Lambda computes behavioural features
> from a DynamoDB feature store, and an XGBoost model on Fargate scores them in under a
> millisecond server-side. The system monitors its own feature drift with PSI and retrains on a
> champion/challenger basis, promoting only if the new model is measurably better. Flagged
> transactions go to an LLM agent on Bedrock that investigates them with tools - account history,
> device, merchant, similar cases - and commits a schema-validated verdict with an analyst
> report. Everything is Terraform and GitHub Actions, and there is an evaluation harness that
> scores the agent against a rule-based baseline on 100 labelled cases.

## Numbers to have ready

* 606k synthetic transactions, 90 days, 15k users, 0.7% fraud across six patterns
* XGBoost: 89.6% recall at 1% FPR on the untouched 14-day test window (rules 66.8% at 5.9% FPR,
  logistic regression 73.7%); PR-AUC 0.88 vs 0.30 for rules; ~96 alerts/day at this volume
* Weakest patterns: velocity bursts and merchant collusion (~61% recall) - both look like normal
  behaviour from a known device
* Travellers are 6x more likely to be false positives than regular users; sale-day shoppers 2.6x
* Scorer: 0.4 ms median server-side, ~1.5 ms when SHAP runs; 717 req/s on two uvicorn workers in
  the load test, p95 63 ms end to end
* Heuristic investigator baseline: decides 57% of cases, 82.5% accurate when it does, 7 false
  blocks and 3 missed frauds out of 100

## Questions you will get, and honest answers

**"The data is synthetic. Why should I believe the numbers?"**
You shouldn't believe the absolute recall - it is a property of the simulator. What transfers is
the methodology: time-based split with burn-in, threshold chosen on validation and reported on
test with the realised FPR, per-pattern recall, false-positive analysis by legitimate behaviour,
and baselines. I also made the simulator adversarial on purpose: victims drawn by activity so
activity does not leak the label, modest fraud multipliers, and legitimate life events that
look like fraud.

**"Why XGBoost and not a neural net / a GNN on the device graph?"**
Tabular, 45 features, 600k rows, sub-millisecond latency budget: gradient boosting is the right
tool and SHAP gives per-decision reasons the analyst and the agent both use. A graph model over
users-devices-merchants is the obvious next step for ring detection; I approximated it with
`device_user_count` and `device_user_age_days`, which are two of the top features.

**"How do you avoid train/serve skew?"**
One `compute_features` function, called by the offline replay and by the Lambda. A test replays
history through both paths and asserts the frames are identical. State is bounded and causal.

**"What happens when Bedrock is slow or down?"**
Flagged transactions queue in SQS FIFO; the investigate Lambda has reserved concurrency 5 and a
DLQ with an alarm. The scoring path does not depend on the agent at all - a flag is a flag whether
or not it gets investigated. Kill switch: switch the backend to the heuristic investigator.

**"Can the LLM block a customer?"**
No. It can only call read tools and `submit_verdict`. The Lambda decides what to do with a
validated verdict, and only `fraud` with confidence >= 0.8 raises an alert. Everything else is
step-up authentication or human review.

**"How do you know the agent is better than rules?"**
The evaluation harness: 100 flagged cases, half fraud (stratified across patterns) and half false
positives (stratified across the behaviours that fooled the model). It reports coverage, decided
accuracy, false blocks, missed fraud, tool calls, tokens and cost per case, and the heuristic
backend runs the same loop with the same tools as the baseline.

**"What would you do with real data?"**
Replace the simulator with the feature log + a label feed (chargebacks, analyst verdicts from the
cases table), move the merchant fraud-rate feature from a static prior to a time-lagged rolling
rate, add a graph feature for device/merchant rings, and calibrate the threshold on the real
alert budget. The pipeline from feature log to retraining is already in place.

**"What was the hardest bug?"**
The first version of the simulator made the model look perfect: 97% recall, logistic regression
nearly as good, and `txn_count_total` as the top SHAP feature - which made no sense. The cause:
victims were sampled uniformly over users while legitimate rows are dominated by active users, so
"low activity" leaked the label. Drawing victims in proportion to activity dropped recall by 7
points and made the feature importances sensible. The lesson: when a model looks too good, audit
the data generation before celebrating. (Reproduce it yourself: replace the weighted victim
sampling in `simulator/generate.py` with `rng.sample(world.users, n_fraud_users)` and retrain.)

## What is deliberately not here

Fine-tuning, a real-time graph database, a UI, multi-region, and a feature store product (Feast).
Each would be a project on its own; the README roadmap lists them so the scope reads as chosen,
not missed.
