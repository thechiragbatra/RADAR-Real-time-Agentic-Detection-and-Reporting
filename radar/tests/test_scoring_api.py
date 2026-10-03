import json
import time
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from radar.features import FEATURE_NAMES
from radar.model.registry import ModelRegistry
from radar.scoring.api import create_app


@pytest.fixture(scope="module")
def client(models_dir: Path):
    app = create_app(str(models_dir))
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module")
def feature_rows(data_dir: Path):
    feats = pd.read_parquet(data_dir / "features.parquet")
    return feats


def test_health_and_model(client):
    h = client.get("/health").json()
    assert h["status"] == "ok" and h["model_version"] == "vtest"
    m = client.get("/model").json()
    assert m["version"] == "vtest" and "feature_stats" not in m and "metrics" in m


def test_score_contract_and_explain_only_when_flagged(client, feature_rows):
    legit = feature_rows[~feature_rows.is_fraud].iloc[0][FEATURE_NAMES].to_dict()
    r = client.post("/score", json={"txn_id": "T1", "features": legit})
    assert r.status_code == 200
    body = r.json()
    assert set(body) >= {
        "txn_id",
        "score",
        "flagged",
        "threshold",
        "model_version",
        "reasons",
        "latency_ms",
    }
    assert 0 <= body["score"] <= 1
    if not body["flagged"]:
        assert body["reasons"] == []
    r2 = client.post("/score", json={"txn_id": "T1", "features": legit, "explain": True}).json()
    assert len(r2["reasons"]) == 3 and {"feature", "value", "contribution", "direction"} <= set(
        r2["reasons"][0]
    )


def test_fraud_rows_score_higher_on_average(client, feature_rows):
    fraud = feature_rows[feature_rows.is_fraud].tail(200)
    legit = feature_rows[~feature_rows.is_fraud].tail(200)
    sf = [
        client.post("/score", json={"txn_id": "x", "features": r}).json()["score"]
        for r in fraud[FEATURE_NAMES].to_dict("records")
    ]
    sl = [
        client.post("/score", json={"txn_id": "x", "features": r}).json()["score"]
        for r in legit[FEATURE_NAMES].to_dict("records")
    ]
    assert sum(sf) / len(sf) > 3 * sum(sl) / len(sl)


def test_batch_and_metrics(client, feature_rows):
    rows = feature_rows.tail(50)[FEATURE_NAMES].to_dict("records")
    r = client.post(
        "/score/batch", json=[{"txn_id": f"T{i}", "features": f} for i, f in enumerate(rows)]
    )
    assert r.status_code == 200 and len(r.json()) == 50
    text = client.get("/metrics").text
    assert "radar_score_requests_total" in text and 'radar_model_info{version="vtest"}' in text


def test_missing_feature_defaults_to_zero_and_bad_payload_rejected(client, feature_rows):
    partial = {"amount": 100.0}
    assert client.post("/score", json={"txn_id": "T", "features": partial}).status_code == 200
    assert client.post("/score", json={"txn_id": "T"}).status_code == 422


def test_hot_swap_on_promotion(models_dir: Path, feature_rows):
    app = create_app(str(models_dir))
    app.state.loader.refresh_seconds = 0
    with TestClient(app) as c:
        assert c.get("/health").json()["model_version"] == "vtest"
        reg = ModelRegistry(str(models_dir))
        reg.publish(models_dir / "vtest", "vtest2")
        meta = json.loads((models_dir / "vtest2" / "metadata.json").read_text())
        meta["version"] = "vtest2"
        (models_dir / "vtest2" / "metadata.json").write_text(json.dumps(meta))
        time.sleep(0.01)
        reg.promote("vtest2", "hot swap test")
        assert c.get("/health").json()["model_version"] == "vtest2"
        reg.promote("vtest", "restore")
        assert c.get("/health").json()["model_version"] == "vtest"
