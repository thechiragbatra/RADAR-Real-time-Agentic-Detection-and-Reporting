"""Scheduled drift monitor.

Every hour: read the feature log written by the stream consumer (Firehose -> S3, JSON
lines partitioned by hour), compute PSI per feature and for the score distribution
against the champion's training reference, publish them to CloudWatch, and raise an
EventBridge ``DriftDetected`` event when the worst feature crosses the threshold. The
EventBridge rule in Terraform runs the retraining task in response.

No labels are needed, which is the point: fraud labels arrive days or weeks later, but a
shift in the input distribution (a new merchant category, a payments outage changing
velocity, a festival) is visible immediately.
"""

from __future__ import annotations

import io
import json
import logging
import time
from datetime import UTC, datetime, timedelta
from typing import Any

import pandas as pd

from radar.config import settings
from radar.model.drift import classify, psi_report
from radar.model.registry import ModelRegistry

log = logging.getLogger("radar.drift_monitor")


def load_feature_log(
    s3, bucket: str, prefix: str, since: datetime, until: datetime
) -> pd.DataFrame:
    """Read JSON-lines objects under hourly prefixes ``<prefix>/YYYY/MM/DD/HH/``."""
    frames = []
    hour = since.replace(minute=0, second=0, microsecond=0)
    while hour <= until:
        p = f"{prefix}/{hour:%Y/%m/%d/%H}/"
        token = None
        while True:
            kw = {"Bucket": bucket, "Prefix": p}
            if token:
                kw["ContinuationToken"] = token
            resp = s3.list_objects_v2(**kw)
            for obj in resp.get("Contents", []):
                body = s3.get_object(Bucket=bucket, Key=obj["Key"])["Body"].read()
                frames.append(pd.read_json(io.BytesIO(body), lines=True))
            token = resp.get("NextContinuationToken")
            if not token:
                break
        hour += timedelta(hours=1)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def evaluate_drift(live: pd.DataFrame, reference: dict[str, dict], threshold: float) -> dict:
    psi = psi_report(reference, live.rename(columns={"score": "__score__"}))
    worst = max(psi.items(), key=lambda kv: kv[1]) if psi else ("none", 0.0)
    shifted = sorted([f for f, v in psi.items() if v >= threshold])
    return {
        "rows": int(len(live)),
        "psi": psi,
        "max_psi": float(worst[1]),
        "max_feature": worst[0],
        "shifted_features": shifted,
        "status": classify(worst[1]),
        "drift_detected": worst[1] >= threshold,
    }


def handler(event: dict, context: Any = None) -> dict:
    import boto3

    region = settings.aws_region
    s3 = boto3.client("s3", region_name=region)
    cw = boto3.client("cloudwatch", region_name=region)
    events = boto3.client("events", region_name=region)

    registry = ModelRegistry(settings.model_uri, region)
    champ = registry.champion()
    if champ is None:
        return {"status": "no_champion"}
    version = champ["version"]
    reference = json.loads(registry.read_text(version, "reference.json") or "{}")

    until = datetime.now(UTC)
    since = until - timedelta(hours=settings.drift_lookback_hours)
    live = load_feature_log(s3, settings.artefact_bucket, "feature-log", since, until)
    if len(live) < 200:
        log.warning("only %d rows in lookback window; skipping", len(live))
        return {"status": "insufficient_data", "rows": int(len(live))}

    report = evaluate_drift(live, reference, settings.psi_alert_threshold)
    report["model_version"] = version

    metric_data = [
        {"MetricName": "FeaturePSI", "Dimensions": [{"Name": "Feature", "Value": f}], "Value": v}
        for f, v in report["psi"].items()
    ]
    metric_data.append(
        {
            "MetricName": "MaxPSI",
            "Dimensions": [{"Name": "ModelVersion", "Value": version}],
            "Value": report["max_psi"],
        }
    )
    metric_data.append(
        {
            "MetricName": "FlagRate",
            "Dimensions": [{"Name": "ModelVersion", "Value": version}],
            "Value": float(
                (live["score"] >= float(champ.get("metrics", {}).get("threshold", 0.5))).mean()
            )
            if "score" in live
            else 0.0,
        }
    )
    for i in range(0, len(metric_data), 20):  # PutMetricData accepts 20 per call
        cw.put_metric_data(Namespace="RADAR", MetricData=metric_data[i : i + 20])

    if report["drift_detected"]:
        events.put_events(
            Entries=[
                {
                    "Source": "radar.drift",
                    "DetailType": "DriftDetected",
                    "Detail": json.dumps(
                        {
                            k: report[k]
                            for k in (
                                "max_psi",
                                "max_feature",
                                "shifted_features",
                                "model_version",
                                "rows",
                            )
                        }
                    ),
                    "Time": datetime.now(UTC),
                }
            ]
        )
        log.warning("drift detected: %s", report["shifted_features"])
    registry.write_text(json.dumps(report, indent=2), "drift", f"{int(time.time())}.json")
    return report
