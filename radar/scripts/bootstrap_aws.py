"""One-time setup after `terraform apply`:

1. upload the trained model bundle(s) to the S3 registry and promote the champion
2. load user/merchant reference data into the DynamoDB state table
3. upload the dataset to S3 so the retraining task can find it

  python scripts/bootstrap_aws.py --bucket <artefact bucket> --table <state table> --models models --data data
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import boto3
import pandas as pd

from radar.features.reference import DynamoDBReferenceData
from radar.model.registry import ModelRegistry


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--bucket", required=True)
    p.add_argument("--table", required=True)
    p.add_argument("--region", default="ap-south-1")
    p.add_argument("--models", type=Path, default=Path("models"))
    p.add_argument("--data", type=Path, default=Path("data"))
    p.add_argument("--skip-reference", action="store_true")
    a = p.parse_args()

    local = ModelRegistry(str(a.models))
    remote = ModelRegistry(f"s3://{a.bucket}/models", a.region)
    champ = local.champion()
    if champ is None:
        raise SystemExit("no local champion - run radar-train --promote first")
    for version in local.list_versions():
        remote.publish(a.models / version, version)
        print(f"uploaded model {version}")
    remote.promote(champ["version"], "bootstrap from local registry", champ.get("metrics"))
    print(f"champion -> {champ['version']}")

    s3 = boto3.client("s3", region_name=a.region)
    for name in ("transactions.parquet", "users.parquet", "merchants.parquet", "meta.json"):
        s3.upload_file(str(a.data / name), a.bucket, f"data/{name}")
        print(f"uploaded data/{name}")

    if not a.skip_reference:
        ddb = boto3.resource("dynamodb", region_name=a.region)
        n = DynamoDBReferenceData.load_from_parquet(
            ddb.Table(a.table),
            pd.read_parquet(a.data / "users.parquet"),
            pd.read_parquet(a.data / "merchants.parquet"),
        )
        print(f"loaded {n} reference items into {a.table}")
    print(json.dumps({"bucket": a.bucket, "champion": champ["version"]}))


if __name__ == "__main__":
    main()
