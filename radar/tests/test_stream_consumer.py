import base64
import json
from pathlib import Path

import boto3
import pandas as pd
import pytest
from moto import mock_aws

from radar.config import settings
from radar.features import InMemoryFeatureStore, LocalReferenceData
from radar.lambdas.common import Deps, LocalScorer
from radar.lambdas.stream_consumer import handler, process_transaction
from radar.schemas import Transaction


def _record(txn: dict, seq: str) -> dict:
    return {
        "kinesis": {
            "sequenceNumber": seq,
            "data": base64.b64encode(json.dumps(txn).encode()).decode(),
        }
    }


@pytest.fixture
def wire_txns(transactions: pd.DataFrame) -> list[dict]:
    fraud_user = transactions[transactions.fraud_type == "card_testing"].user_id.iloc[0]
    sub = transactions[transactions.user_id == fraud_user].head(30)
    cols = list(Transaction.model_fields)
    out = []
    for row in sub[cols].itertuples(index=False):
        d = dict(zip(cols, row, strict=True))
        d["is_online"] = bool(d["is_online"])
        out.append(d)
    return out


@mock_aws
def test_consumer_end_to_end(data_dir: Path, models_dir: Path, wire_txns, monkeypatch):
    ddb = boto3.resource("dynamodb", region_name="ap-south-1")
    ddb.create_table(
        TableName="decisions",
        KeySchema=[{"AttributeName": "txn_id", "KeyType": "HASH"}],
        AttributeDefinitions=[{"AttributeName": "txn_id", "AttributeType": "S"}],
        BillingMode="PAY_PER_REQUEST",
    )
    sqs = boto3.client("sqs", region_name="ap-south-1")
    q = sqs.create_queue(
        QueueName="inv.fifo", Attributes={"FifoQueue": "true", "ContentBasedDeduplication": "false"}
    )["QueueUrl"]
    monkeypatch.setattr(settings, "investigation_queue_url", q)
    monkeypatch.setattr(settings, "firehose_feature_log", "")

    deps = Deps(
        store=InMemoryFeatureStore(),
        ref=LocalReferenceData.from_dir(data_dir),
        scorer=LocalScorer(str(models_dir)),
        dynamodb=ddb,
        sqs=sqs,
        decisions_table=ddb.Table("decisions"),
    )
    import radar.lambdas.stream_consumer as sc

    monkeypatch.setattr(sc, "get_deps", lambda: deps)

    event = {
        "Records": [_record(t, str(i)) for i, t in enumerate(wire_txns)]
        + [_record({"garbage": True}, "bad")]
    }
    out = handler(event, None)
    assert out["batchItemFailures"] == [{"itemIdentifier": "bad"}]

    items = ddb.Table("decisions").scan()["Items"]
    assert len(items) == len(wire_txns)
    item = next(i for i in items if i["txn_id"] == wire_txns[-1]["txn_id"])
    assert set(item) >= {"score", "flagged", "features", "txn", "model_version", "expires_at"}
    assert (
        json.loads(item["features"])["txn_count_total"] == len(wire_txns) - 1
    )  # state was updated per record

    flagged = [i for i in items if i["flagged"]]
    msgs = sqs.receive_message(QueueUrl=q, MaxNumberOfMessages=10).get("Messages", [])
    assert len(flagged) > 0 and len(msgs) == min(10, len(flagged))
    body = json.loads(msgs[0]["Body"])
    assert body["txn_id"] in {i["txn_id"] for i in flagged}


def test_process_transaction_without_aws_clients(data_dir: Path, models_dir: Path, wire_txns):
    deps = Deps(
        store=InMemoryFeatureStore(),
        ref=LocalReferenceData.from_dir(data_dir),
        scorer=LocalScorer(str(models_dir)),
    )
    d = process_transaction(Transaction(**wire_txns[0]), deps)
    assert {"score", "flagged", "features", "reasons"} <= set(d)
    assert d["features"]["txn_count_total"] == 0
