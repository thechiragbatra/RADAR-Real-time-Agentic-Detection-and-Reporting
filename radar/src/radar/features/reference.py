"""Static reference data: user and merchant profiles.

In production these come from the bank's customer and merchant systems; here they are
the simulator's ``users.parquet`` / ``merchants.parquet`` (local) or a DynamoDB table
loaded from them (AWS).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol

import pandas as pd

from radar.schemas import MerchantProfile, UserProfile


class ReferenceData(Protocol):
    def get_user_profile(self, user_id: str) -> UserProfile | None: ...
    def get_merchant(self, merchant_id: str) -> MerchantProfile | None: ...


class LocalReferenceData:
    def __init__(self, users: pd.DataFrame, merchants: pd.DataFrame) -> None:
        self._users = {
            r.user_id: UserProfile(
                user_id=r.user_id,
                home_city=r.home_city,
                home_lat=float(r.home_lat),
                home_lon=float(r.home_lon),
                account_created_ts=float(r.account_created_ts),
                spend_level=float(r.spend_level),
            )
            for r in users.itertuples(index=False)
        }
        self._merchants = {
            r.merchant_id: MerchantProfile(
                merchant_id=r.merchant_id,
                category=r.category,
                city=r.city,
                is_online=bool(r.is_online),
                created_ts=float(r.created_ts),
                category_risk=float(r.category_risk),
            )
            for r in merchants.itertuples(index=False)
        }

    @classmethod
    def from_dir(cls, data_dir: str | Path) -> LocalReferenceData:
        d = Path(data_dir)
        return cls(pd.read_parquet(d / "users.parquet"), pd.read_parquet(d / "merchants.parquet"))

    def get_user_profile(self, user_id: str) -> UserProfile | None:
        return self._users.get(user_id)

    def get_merchant(self, merchant_id: str) -> MerchantProfile | None:
        return self._merchants.get(merchant_id)


class DynamoDBReferenceData:
    """Same single-table layout as the feature store: ``pk`` = ``PROFILE#<id>`` / ``MERCHANT#<id>``."""

    def __init__(self, table_name: str, region: str | None = None, resource=None) -> None:
        if resource is None:
            import boto3

            resource = boto3.resource("dynamodb", region_name=region)
        self.table = resource.Table(table_name)
        self._cache: dict[str, dict | None] = {}

    def _get(self, pk: str) -> dict | None:
        if pk in self._cache:
            return self._cache[pk]
        item = self.table.get_item(Key={"pk": pk}).get("Item")
        doc = json.loads(item["doc"]) if item else None
        self._cache[pk] = doc
        return doc

    def get_user_profile(self, user_id: str) -> UserProfile | None:
        d = self._get(f"PROFILE#{user_id}")
        return UserProfile(**d) if d else None

    def get_merchant(self, merchant_id: str) -> MerchantProfile | None:
        d = self._get(f"MERCHANT#{merchant_id}")
        return MerchantProfile(**d) if d else None

    @staticmethod
    def load_from_parquet(table, users: pd.DataFrame, merchants: pd.DataFrame) -> int:
        """Bulk-load reference data (used by scripts/bootstrap_aws.py)."""
        n = 0
        with table.batch_writer() as bw:
            for r in users.itertuples(index=False):
                bw.put_item(Item={"pk": f"PROFILE#{r.user_id}", "doc": json.dumps(r._asdict())})
                n += 1
            for r in merchants.itertuples(index=False):
                doc = r._asdict()
                doc["is_online"] = bool(doc["is_online"])
                bw.put_item(Item={"pk": f"MERCHANT#{r.merchant_id}", "doc": json.dumps(doc)})
                n += 1
        return n
