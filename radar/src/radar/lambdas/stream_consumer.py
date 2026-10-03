"""Kinesis consumer: the online scoring path.

For every record: parse -> compute features from the user's rolling state -> score ->
persist the decision -> log the feature vector for drift monitoring -> enqueue flagged
transactions for investigation -> update state.

Ordering: Kinesis delivers a user's records in order (partition key = user_id) and this
handler processes a batch sequentially, so the feature store is updated exactly once per
transaction with no locking. Failures are reported per record
(``ReportBatchItemFailures``) so one bad record does not block the shard.
"""

from __future__ import annotations

import base64
import json
import logging
import time
from decimal import Decimal
from typing import Any

from radar.config import settings
from radar.features import featurize_and_update
from radar.lambdas.common import Deps, emf, get_deps
from radar.schemas import Transaction

log = logging.getLogger("radar.stream_consumer")
DECISION_TTL_SECONDS = 30 * 86400


def _to_dynamo(obj: Any) -> Any:
    """DynamoDB rejects floats; convert recursively to Decimal via str."""
    if isinstance(obj, float):
        return Decimal(str(round(obj, 6)))
    if isinstance(obj, dict):
        return {k: _to_dynamo(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_dynamo(v) for v in obj]
    return obj


def process_transaction(txn: Transaction, deps: Deps) -> dict:
    t0 = time.perf_counter()
    features = featurize_and_update(txn, deps.store, deps.ref)
    result = deps.scorer.score(txn.txn_id, features)
    decision = {
        "txn_id": txn.txn_id,
        "ts": txn.ts,
        "user_id": txn.user_id,
        "merchant_id": txn.merchant_id,
        "device_id": txn.device_id,
        "amount": txn.amount,
        "score": result.score,
        "flagged": result.flagged,
        "threshold": result.threshold,
        "model_version": result.model_version,
        "reasons": [r.model_dump() for r in result.reasons],
        "features": features,
        "txn": txn.model_dump(mode="json"),
        "scored_at": time.time(),
    }

    if deps.decisions_table is not None:
        item = _to_dynamo(
            {**decision, "features": json.dumps(features), "txn": json.dumps(decision["txn"])}
        )
        item["expires_at"] = int(time.time()) + DECISION_TTL_SECONDS
        deps.decisions_table.put_item(Item=item)

    if deps.firehose is not None and settings.firehose_feature_log:
        line = (
            json.dumps(
                {
                    "txn_id": txn.txn_id,
                    "ts": txn.ts,
                    "score": result.score,
                    "model_version": result.model_version,
                    **features,
                }
            )
            + "\n"
        )
        deps.firehose.put_record(
            DeliveryStreamName=settings.firehose_feature_log, Record={"Data": line.encode()}
        )

    if result.flagged and deps.sqs is not None and settings.investigation_queue_url:
        deps.sqs.send_message(
            QueueUrl=settings.investigation_queue_url,
            MessageBody=json.dumps(
                {
                    "txn_id": txn.txn_id,
                    "user_id": txn.user_id,
                    "score": result.score,
                    "model_version": result.model_version,
                }
            ),
            MessageGroupId=txn.user_id,
            MessageDeduplicationId=txn.txn_id,
        )

    emf(
        {
            "ScoredTransactions": 1,
            "Flagged": int(result.flagged),
            "ScoreLatency": result.latency_ms,
            "EndToEndLatency": (time.perf_counter() - t0) * 1000,
        },
        {"ModelVersion": result.model_version},
    )
    return decision


def parse_record(record: dict) -> Transaction:
    payload = base64.b64decode(record["kinesis"]["data"])
    return Transaction.model_validate_json(payload)


def handler(event: dict, context: Any = None) -> dict:
    deps = get_deps()
    failures = []
    for record in event.get("Records", []):
        try:
            process_transaction(parse_record(record), deps)
        except Exception:
            log.exception("failed record %s", record.get("kinesis", {}).get("sequenceNumber"))
            failures.append({"itemIdentifier": record["kinesis"]["sequenceNumber"]})
    return {"batchItemFailures": failures}
