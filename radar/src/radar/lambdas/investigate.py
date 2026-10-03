"""SQS consumer: runs the investigation agent on flagged transactions.

Flow: SQS (FIFO, grouped by user) -> Investigator (Bedrock) -> cases table -> SNS alert
when the verdict is fraud with confidence >= 0.8. The Lambda, not the model, decides what
happens to the account: the model only produces a validated ``InvestigationResult``.

Reserved concurrency is kept low (Terraform: 5) because Bedrock throughput, not
Lambda, is the bottleneck, and a flood of alerts should queue rather than fail.
"""

from __future__ import annotations

import json
import logging
import time
from decimal import Decimal
from functools import lru_cache
from io import BytesIO
from typing import Any

import pandas as pd

from radar.agent.datasource import DynamoCaseDataSource
from radar.agent.investigator import Investigator
from radar.agent.llm import BedrockBackend, HeuristicBackend
from radar.agent.report import render_markdown
from radar.config import settings
from radar.lambdas.common import Deps, emf, get_deps
from radar.model.registry import ModelRegistry

log = logging.getLogger("radar.investigate")
BLOCK_CONFIDENCE = 0.8


@lru_cache(maxsize=1)
def _investigator() -> Investigator:
    deps = get_deps()
    registry = ModelRegistry(settings.model_uri, settings.aws_region)
    champ = registry.champion()
    if champ is None:
        raise RuntimeError("no champion model")
    version = champ["version"]
    meta = registry.metadata(version) or {}
    if registry.is_s3:
        obj = registry._s3.get_object(
            Bucket=registry.bucket, Key=registry._key(version, "case_bank.parquet")
        )
        bank = pd.read_parquet(BytesIO(obj["Body"].read()))
    else:
        bank = pd.read_parquet(f"{registry.root}/{version}/case_bank.parquet")
    source = DynamoCaseDataSource(
        deps.dynamodb,
        settings.decisions_table,
        settings.user_state_table,
        bank,
        meta["feature_stats"],
    )
    backend = HeuristicBackend() if settings.bedrock_model_id == "heuristic" else BedrockBackend()
    return Investigator(backend, source)


def _to_dynamo(obj: Any) -> Any:
    if isinstance(obj, float):
        return Decimal(str(round(obj, 6)))
    if isinstance(obj, dict):
        return {k: _to_dynamo(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_dynamo(v) for v in obj]
    return obj


def investigate_one(txn_id: str, deps: Deps, inv: Investigator | None = None) -> dict:
    inv = inv or _investigator()
    trace = inv.run(txn_id)
    report = render_markdown(trace)
    result = trace.result
    record = {
        "case_id": trace.case_id,
        "txn_id": txn_id,
        "verdict": result.verdict.value,
        "confidence": result.confidence,
        "fraud_type": result.fraud_type.value,
        "recommended_action": result.recommended_action.value,
        "key_evidence": result.key_evidence,
        "analyst_summary": result.analyst_summary,
        "report_md": report,
        "trace": json.dumps(trace.model_dump(mode="json")),
        "tokens": trace.input_tokens + trace.output_tokens,
        "cost_usd": trace.cost_usd,
        "latency_ms": trace.latency_ms,
        "terminated_by": trace.terminated_by,
        "created_at": time.time(),
    }
    if deps.cases_table is not None:
        deps.cases_table.put_item(Item=_to_dynamo(record))

    should_alert = result.verdict.value == "fraud" and result.confidence >= BLOCK_CONFIDENCE
    if should_alert and deps.sns is not None and settings.alerts_topic_arn:
        deps.sns.publish(
            TopicArn=settings.alerts_topic_arn,
            Subject=f"[RADAR] probable fraud {txn_id} ({result.confidence:.0%})",
            Message=report,
        )
    emf(
        {
            "Investigations": 1,
            "VerdictFraud": int(result.verdict.value == "fraud"),
            "VerdictLegit": int(result.verdict.value == "legit"),
            "VerdictReview": int(result.verdict.value == "needs_review"),
            "Alerts": int(should_alert),
            "AgentTokens": trace.input_tokens + trace.output_tokens,
            "AgentLatency": trace.latency_ms,
            "Incomplete": int(trace.terminated_by != "submit_verdict"),
        },
        {"Backend": trace.backend},
    )
    return record


def handler(event: dict, context: Any = None) -> dict:
    deps = get_deps()
    failures = []
    for msg in event.get("Records", []):
        try:
            body = json.loads(msg["body"])
            investigate_one(body["txn_id"], deps)
        except Exception:
            log.exception("investigation failed for message %s", msg.get("messageId"))
            failures.append({"itemIdentifier": msg["messageId"]})
    return {"batchItemFailures": failures}
