| metric | heuristic |
|---|---|
| coverage | 57.0% |
| decided_accuracy | 82.5% |
| fraud_recall | 60.0% |
| fraud_precision | 81.1% |
| false_blocks | 7 |
| missed_fraud | 3 |
| fraud_type_accuracy | 33.3% |
| avg_tool_calls | 7.0 |
| p50_latency_ms | 40 |
| avg_tokens | 0.0 |
| avg_cost_usd | $0.0000 |
| monthly_cost_usd_at_100_alerts_per_day | $0 |

### Decided accuracy by stratum

| stratum | n | heuristic |
|---|---|---|
| account_takeover | 9 | 100% (cov 89%) |
| card_testing | 9 | 100% (cov 100%) |
| geo_jump | 8 | 67% (cov 38%) |
| legit:big_purchase | 3 | 0% (cov 67%) |
| legit:phone_upgrade | 11 | 100% (cov 27%) |
| legit:regular | 12 | 60% (cov 42%) |
| legit:sale_spree | 12 | 25% (cov 33%) |
| legit:trip | 12 | 100% (cov 83%) |
| low_and_slow | 8 | 100% (cov 100%) |
| merchant_collusion | 8 | 100% (cov 38%) |
| velocity_burst | 8 | 0% (cov 25%) |
