"""Shared plumbing for the Lambda handlers: cached clients, scorer client, EMF metrics."""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import httpx

from radar.config import settings
from radar.features import DynamoDBFeatureStore, FeatureStore, InMemoryFeatureStore, ReferenceData
from radar.features.reference import DynamoDBReferenceData, LocalReferenceData
from radar.schemas import Reason, ScoreResponse

log = logging.getLogger("radar.lambdas")
logging.basicConfig(level=settings.log_level)


# ---- scorer clients -------------------------------------------------------------------
class HttpScorer:
    """Calls the Fargate scoring service. One keep-alive client per warm Lambda."""

    def __init__(self, base_url: str, timeout: float, api_key: str = "") -> None:
        headers = {"x-radar-key": api_key} if api_key else {}
        self.client = httpx.Client(base_url=base_url, timeout=timeout, headers=headers)

    def score(self, txn_id: str, features: dict[str, float]) -> ScoreResponse:
        r = self.client.post("/score", json={"txn_id": txn_id, "features": features})
        r.raise_for_status()
        return ScoreResponse.model_validate(r.json())


class LocalScorer:
    """Scores in-process from a model registry. Used in tests and for the
    'model inside the Lambda' deployment variant (fewer hops, no ALB)."""

    def __init__(self, registry_uri: str) -> None:
        from radar.scoring.scorer import ChampionLoader

        self.loader = ChampionLoader(registry_uri, refresh_seconds=300)

    def score(self, txn_id: str, features: dict[str, float]) -> ScoreResponse:
        t0 = time.perf_counter()
        b = self.loader.get()
        s = b.score(features)
        flagged = s >= b.threshold
        reasons: list[Reason] = b.explain(features) if flagged else []
        return ScoreResponse(
            txn_id=txn_id,
            score=s,
            flagged=flagged,
            threshold=b.threshold,
            model_version=b.version,
            reasons=reasons,
            latency_ms=(time.perf_counter() - t0) * 1000,
        )


# ---- dependency bundle -----------------------------------------------------------
@dataclass
class Deps:
    store: FeatureStore
    ref: ReferenceData
    scorer: Any
    dynamodb: Any = None
    sqs: Any = None
    firehose: Any = None
    sns: Any = None
    decisions_table: Any = None
    cases_table: Any = None


@lru_cache(maxsize=1)
def get_deps() -> Deps:
    """Build dependencies from the environment. ``RADAR_ENV=local`` wires in-memory
    stores and a local model so the handlers can run on a laptop."""
    if settings.env == "local":
        data_dir = Path(settings.data_dir)
        ref = LocalReferenceData.from_dir(data_dir)
        return Deps(store=InMemoryFeatureStore(), ref=ref, scorer=LocalScorer(settings.model_uri))

    import boto3

    region = settings.aws_region
    dynamodb = boto3.resource("dynamodb", region_name=region)
    store = DynamoDBFeatureStore(settings.user_state_table, resource=dynamodb)
    ref = DynamoDBReferenceData(
        os.environ.get("RADAR_REFERENCE_TABLE", settings.user_state_table), resource=dynamodb
    )
    scorer = (
        LocalScorer(settings.model_uri)
        if settings.scorer_url == "local"
        else HttpScorer(
            settings.scorer_url, settings.scorer_timeout_seconds, settings.scorer_api_key
        )
    )
    return Deps(
        store=store,
        ref=ref,
        scorer=scorer,
        dynamodb=dynamodb,
        sqs=boto3.client("sqs", region_name=region),
        firehose=boto3.client("firehose", region_name=region),
        sns=boto3.client("sns", region_name=region),
        decisions_table=dynamodb.Table(settings.decisions_table),
        cases_table=dynamodb.Table(settings.cases_table),
    )


# ---- CloudWatch Embedded Metric Format ----------------------------------------------
def emf(
    metrics: dict[str, float], dimensions: dict[str, str] | None = None, namespace: str = "RADAR"
) -> None:
    """Emit metrics by printing a structured log line - no PutMetricData API call, no
    extra latency on the hot path. CloudWatch parses the ``_aws`` block."""
    if settings.env == "local":  # keep local runs readable
        return
    dims = dimensions or {}
    doc = {
        "_aws": {
            "Timestamp": int(time.time() * 1000),
            "CloudWatchMetrics": [
                {
                    "Namespace": namespace,
                    "Dimensions": [list(dims.keys())],
                    "Metrics": [
                        {"Name": k, "Unit": "Milliseconds" if k.endswith("Latency") else "Count"}
                        for k in metrics
                    ],
                }
            ],
        },
        **dims,
        **metrics,
    }
    print(json.dumps(doc))
