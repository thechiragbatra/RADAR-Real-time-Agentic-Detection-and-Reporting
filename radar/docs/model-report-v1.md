# Model report - version v1

Trained 2026-10-03T17:20:18.835234+00:00. Rows: {'train': 357452, 'val': 58376, 'test': 81332}. Target FPR 1.0% (threshold 0.1211 chosen on validation, metrics below are on the untouched test window).

| model | ROC-AUC | PR-AUC | recall@0.5%FPR | recall@1%FPR | recall@2%FPR | deployed op. point | precision | alerts/day |
|---|---|---|---|---|---|---|---|---|
| rules | 0.814 | 0.301 | 66.8% | 66.8% | 66.8% | 66.8% @ 5.89% FPR | 7.9% | 369 |
| logistic_regression | 0.956 | 0.676 | 69.4% | 73.7% | 77.0% | 73.7% @ 0.96% FPR | 36.7% | 87 |
| xgboost | 0.982 | 0.882 | 88.0% | 89.6% | 91.6% | 89.6% @ 0.99% FPR | 40.6% | 96 |

## Recall by fraud pattern (XGBoost, deployed threshold)

| pattern | n | recall |
|---|---|---|
| account_takeover | 90 | 100.0% |
| card_testing | 272 | 100.0% |
| geo_jump | 37 | 100.0% |
| low_and_slow | 48 | 97.9% |
| merchant_collusion | 34 | 61.8% |
| velocity_burst | 127 | 61.4% |

## Top features by mean |SHAP|

| feature | mean abs SHAP |
|---|---|
| device_user_age_days | 0.8898 |
| seconds_since_last | 0.4487 |
| hour_local | 0.3123 |
| speed_kmh | 0.2523 |
| amount | 0.2356 |
| channel_card | 0.2104 |
| device_age_days | 0.2009 |
| is_online | 0.1936 |
| txn_count_1h | 0.1886 |
| amount_ratio_mean | 0.1325 |
| category_risk | 0.1309 |
| amount_sum_24h | 0.1258 |
| is_home_city | 0.1227 |
| txn_count_total | 0.1049 |
| cat_grocery | 0.0951 |
