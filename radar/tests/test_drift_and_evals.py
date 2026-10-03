import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import boto3
import numpy as np
import pandas as pd
from moto import mock_aws

from radar.agent.investigator import Investigator
from radar.agent.llm import HeuristicBackend
from radar.agent.local import build_local_source
from radar.evals.harness import build_cases, render_compare, score_run
from radar.features import FEATURE_NAMES
from radar.lambdas.drift_monitor import evaluate_drift, load_feature_log


def test_evaluate_drift_on_live_window(data_dir: Path, model_dir: Path):
    reference = json.loads((model_dir / "reference.json").read_text())
    feats = pd.read_parquet(data_dir / "features.parquet")
    meta = json.loads((model_dir / "metadata.json").read_text())
    live = feats[feats.ts >= meta["window"]["test"][0]].copy()
    live["score"] = 0.01  # score column is renamed to __score__ internally
    rep = evaluate_drift(live, reference, threshold=0.2)
    assert rep["rows"] == len(live) and set(rep["psi"]) >= set(FEATURE_NAMES)
    assert rep["max_feature"] in rep["psi"]
    # same simulator, next two weeks: features are stable; the fake constant score is not
    stable = [f for f in FEATURE_NAMES if rep["psi"][f] < 0.2]
    assert len(stable) > 0.8 * len(FEATURE_NAMES)
    assert rep["psi"]["__score__"] > 0.2 and rep["drift_detected"]

    shifted = live.copy()
    shifted["amount"] *= 5
    rep2 = evaluate_drift(shifted.drop(columns=["score"]), reference, threshold=0.2)
    assert "amount" in rep2["shifted_features"]


@mock_aws
def test_load_feature_log_reads_hourly_prefixes():
    s3 = boto3.client("s3", region_name="ap-south-1")
    s3.create_bucket(
        Bucket="radar-test", CreateBucketConfiguration={"LocationConstraint": "ap-south-1"}
    )
    now = datetime(2026, 7, 1, 12, 30, tzinfo=UTC)
    for h in (2, 1):
        t = now - timedelta(hours=h)
        body = "\n".join(
            json.dumps({"txn_id": f"T{h}{i}", "amount": float(i), "score": 0.1}) for i in range(5)
        )
        s3.put_object(
            Bucket="radar-test",
            Key=f"feature-log/{t:%Y/%m/%d/%H}/part-{h}.json",
            Body=body.encode(),
        )
    df = load_feature_log(s3, "radar-test", "feature-log", now - timedelta(hours=3), now)
    assert len(df) == 10 and {"txn_id", "amount", "score"} <= set(df.columns)
    assert load_feature_log(
        s3, "radar-test", "feature-log", now + timedelta(hours=5), now + timedelta(hours=6)
    ).empty


def test_build_cases_is_balanced_and_stratified(data_dir: Path, model_dir: Path):
    cases = build_cases(data_dir, model_dir, n=16, seed=1)
    assert 0 < len(cases) <= 16 and cases.txn_id.is_unique
    assert cases.is_fraud.sum() >= 1 and (~cases.is_fraud).sum() >= 1  # both classes present
    assert cases.is_fraud.sum() <= 8  # never more than half fraud
    assert cases.stratum.nunique() >= 3


def test_score_run_metrics(data_dir: Path, model_dir: Path):
    cases = build_cases(data_dir, model_dir, n=12, seed=2)
    src, _ = build_local_source(data_dir, model_dir)
    inv = Investigator(HeuristicBackend(), src)
    traces = [inv.run(t) for t in cases.txn_id]
    res = score_run(cases, traces)
    assert res["n"] == 12 and 0 <= res["coverage"] <= 1
    assert res["false_blocks"] + res["missed_fraud"] <= 12
    assert sum(res["confusion"].values()) == 12
    md = render_compare({"heuristic": res, "other": res})
    assert "| coverage |" in md and "Decided accuracy by stratum" in md
    assert np.isclose(res["avg_tool_calls"], 7.0)
