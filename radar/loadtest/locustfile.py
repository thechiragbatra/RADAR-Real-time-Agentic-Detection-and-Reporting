"""Load test for the scoring service.

    locust -f loadtest/locustfile.py --host http://localhost:8000 --headless \
        -u 50 -r 10 -t 2m --csv loadtest/results/local

Against AWS add the ALB header:  RADAR_SCORER_API_KEY=... locust ... --host http://<alb-dns>

Feature vectors are sampled from data/features.parquet so the request mix (and the share of
flagged transactions, which trigger SHAP) matches production. Read p95 from the Locust
summary; the ECS autoscaling target (300 req/min per task) is set so that a 50-user run
scales the service from 1 to 3 tasks within ~3 minutes.
"""

from __future__ import annotations

import os
import random
from pathlib import Path

import pandas as pd
from locust import HttpUser, between, task

FEATURES = Path(os.environ.get("RADAR_FEATURES", "data/features.parquet"))
API_KEY = os.environ.get("RADAR_SCORER_API_KEY", "")


def _load_rows(n: int = 5000) -> list[dict]:
    from radar.features import FEATURE_NAMES

    if not FEATURES.exists():
        raise SystemExit(f"{FEATURES} not found - run radar-simulate and radar-train first")
    df = pd.read_parquet(FEATURES).sample(n, random_state=0)
    return df[FEATURE_NAMES].to_dict("records")


ROWS = _load_rows()


class ScoringUser(HttpUser):
    wait_time = between(0.01, 0.05)

    def on_start(self) -> None:
        if API_KEY:
            self.client.headers["x-radar-key"] = API_KEY

    @task(10)
    def score(self) -> None:
        row = random.choice(ROWS)
        with self.client.post(
            "/score", json={"txn_id": "load", "features": row}, name="/score", catch_response=True
        ) as r:
            if r.status_code != 200:
                r.failure(f"status {r.status_code}")
            elif r.json()["latency_ms"] > 100:
                r.failure("server-side latency > 100 ms")

    @task(1)
    def health(self) -> None:
        self.client.get("/health", name="/health")
