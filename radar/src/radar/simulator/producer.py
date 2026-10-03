"""Replay a transaction file onto the Kinesis stream.

    radar-produce --file data/transactions.parquet --rate 200 --limit 50000

Labels are stripped before sending: the stream carries exactly what a payment gateway
would emit. Partition key = user_id, which guarantees per-user ordering and lets the
consumer update a user's feature state without locking.
"""

from __future__ import annotations

import argparse
import json
import sys
import time

import pandas as pd

from radar.config import settings
from radar.schemas import Transaction

WIRE_FIELDS = list(Transaction.model_fields)


def iter_records(df: pd.DataFrame):
    for row in df[WIRE_FIELDS].itertuples(index=False):
        yield dict(zip(WIRE_FIELDS, row, strict=True))


def main() -> None:
    p = argparse.ArgumentParser(description="Replay transactions onto Kinesis")
    p.add_argument("--file", default="data/transactions.parquet")
    p.add_argument("--stream", default=settings.kinesis_stream)
    p.add_argument("--rate", type=float, default=100.0, help="records per second")
    p.add_argument("--limit", type=int, default=0, help="0 = all")
    p.add_argument("--offset", type=int, default=0)
    p.add_argument("--dry-run", action="store_true", help="print instead of sending")
    a = p.parse_args()

    df = pd.read_parquet(a.file)
    if a.offset:
        df = df.iloc[a.offset :]
    if a.limit:
        df = df.iloc[: a.limit]

    client = None
    if not a.dry_run:
        import boto3

        client = boto3.client("kinesis", region_name=settings.aws_region)

    batch: list[dict] = []
    sent = 0
    started = time.time()
    for rec in iter_records(df):
        rec["is_online"] = bool(rec["is_online"])
        batch.append({"Data": json.dumps(rec).encode(), "PartitionKey": rec["user_id"]})
        if len(batch) == 100:
            sent += _flush(client, a.stream, batch, a.dry_run)
            batch = []
            # crude rate limiter
            expected = sent / a.rate
            elapsed = time.time() - started
            if expected > elapsed:
                time.sleep(expected - elapsed)
    if batch:
        sent += _flush(client, a.stream, batch, a.dry_run)
    print(f"sent {sent} records in {time.time() - started:.1f}s", file=sys.stderr)


def _flush(client, stream: str, batch: list[dict], dry_run: bool) -> int:
    if dry_run:
        for b in batch[:3]:
            print(b["Data"].decode())
        return len(batch)
    resp = client.put_records(StreamName=stream, Records=batch)
    failed = resp.get("FailedRecordCount", 0)
    if failed:
        retry = [b for b, r in zip(batch, resp["Records"], strict=True) if "ErrorCode" in r]
        time.sleep(0.5)
        return len(batch) - failed + _flush(client, stream, retry, dry_run)
    return len(batch)


if __name__ == "__main__":
    main()
