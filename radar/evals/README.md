# Agent evaluation

```
make eval-cases          # 100 flagged transactions from the test window: 50 fraud, 50 false positives
make eval-heuristic      # rule-based investigator (baseline, free, deterministic)
make eval-bedrock        # LLM agent via Amazon Bedrock (needs AWS credentials)
make eval-compare        # side-by-side table -> evals/results/comparison.md
```

`cases/cases.jsonl` is committed so runs are comparable. Each run writes
`results/<label>.json` (metrics + per-case rows) and `results/<label>.traces.jsonl` (full tool
traces, git-ignored because they are large).

## How to read the comparison

* **coverage** - share of cases the investigator committed to (fraud/legit) instead of `needs_review`.
  Low coverage is not failure: a cautious investigator hands ambiguous cases to a human.
* **decided_accuracy** - accuracy on committed cases. Read together with coverage.
* **false_blocks** - legitimate customers that would have been blocked (verdict fraud, confidence >= 0.8).
  The most expensive mistake; the baseline makes 7 per 100.
* **missed_fraud** - fraud released as legit.
* **by stratum** - which fraud patterns and which legitimate behaviours each investigator handles.
  The baseline is strong on account takeover / card testing (device signals) and weak on velocity
  bursts and sale-day sprees, which look identical without judgement about *what* was bought.

The LLM column is empty until you run it with your own AWS account. For it to be worth deploying it
should raise coverage *and* decided accuracy while reducing false blocks, at a cost per case that
is small next to an analyst's time (~Rs 50-100 per manual review).

## Current results

See [results/comparison.md](results/comparison.md).
