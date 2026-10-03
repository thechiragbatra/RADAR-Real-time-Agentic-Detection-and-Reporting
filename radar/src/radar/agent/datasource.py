"""Where the agent's tools get their facts.

``LocalCaseDataSource`` serves the simulator's files (used by tests, the CLI and the
evaluation harness). ``DynamoCaseDataSource`` reads the live tables written by the
stream consumer. Both expose the same read-only surface; the agent never writes.

Leakage rule for the local source: anything derived from labels (merchant fraud rate,
the similar-case bank) only uses transactions before ``label_cutoff_ts`` - i.e. cases
that an analyst would already have resolved by the time the new one is investigated.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import pandas as pd

from radar.features import FEATURE_NAMES

TXN_PUBLIC_FIELDS = [
    "txn_id", "ts", "user_id", "merchant_id", "merchant_category", "device_id", "channel",
    "amount", "city", "country", "is_online",
]  # fmt: skip


class CaseDataSource(Protocol):
    def transaction(self, txn_id: str) -> dict | None: ...
    def decision(self, txn_id: str) -> dict | None: ...
    def user_profile(self, user_id: str) -> dict | None: ...
    def user_transactions(
        self, user_id: str, before_ts: float, hours: float, limit: int
    ) -> list[dict]: ...
    def merchant_profile(self, merchant_id: str, before_ts: float) -> dict | None: ...
    def device_history(self, device_id: str, before_ts: float) -> dict | None: ...
    def similar_cases(self, txn_id: str, k: int) -> list[dict]: ...


def _clean(d: dict) -> dict:
    out = {}
    for k, v in d.items():
        if isinstance(v, np.generic):
            v = v.item()
        if isinstance(v, float):
            v = round(v, 4)
        out[k] = v
    return out


class LocalCaseDataSource:
    def __init__(
        self,
        transactions: pd.DataFrame,
        features: pd.DataFrame,
        users: pd.DataFrame,
        merchants: pd.DataFrame,
        case_bank: pd.DataFrame,
        feature_stats: dict[str, dict[str, float]],
        label_cutoff_ts: float,
        decisions: pd.DataFrame | None = None,
    ) -> None:
        self.txns = transactions.set_index("txn_id", drop=False)
        self.features = features.set_index("txn_id", drop=False)
        self.users = users.set_index("user_id", drop=False)
        self.merchants = merchants.set_index("merchant_id", drop=False)
        self.label_cutoff_ts = label_cutoff_ts
        self.decisions = (
            decisions.set_index("txn_id", drop=False) if decisions is not None else None
        )
        self._by_user = transactions.sort_values("ts").groupby("user_id")
        self._by_device = transactions.groupby("device_id")
        self._by_merchant = transactions[transactions.ts < label_cutoff_ts].groupby("merchant_id")
        # standardised case bank for nearest-neighbour search
        self.bank = case_bank[case_bank.ts < label_cutoff_ts].reset_index(drop=True)
        self._mu = np.array([feature_stats[f]["mean"] for f in FEATURE_NAMES])
        self._sd = np.array([feature_stats[f]["std"] or 1.0 for f in FEATURE_NAMES])
        bank_x = (self.bank[FEATURE_NAMES].to_numpy() - self._mu) / self._sd
        self._bank_x = bank_x / (np.linalg.norm(bank_x, axis=1, keepdims=True) + 1e-9)

    @classmethod
    def from_dirs(
        cls,
        data_dir: str | Path,
        model_dir: str | Path,
        label_cutoff_ts: float,
        decisions: pd.DataFrame | None = None,
    ) -> LocalCaseDataSource:
        d, m = Path(data_dir), Path(model_dir)
        meta = json.loads((m / "metadata.json").read_text())
        return cls(
            pd.read_parquet(d / "transactions.parquet"),
            pd.read_parquet(d / "features.parquet"),
            pd.read_parquet(d / "users.parquet"),
            pd.read_parquet(d / "merchants.parquet"),
            pd.read_parquet(m / "case_bank.parquet"),
            meta["feature_stats"],
            label_cutoff_ts,
            decisions,
        )

    # ---- reads ---------------------------------------------------------------------
    def transaction(self, txn_id: str) -> dict | None:
        if txn_id not in self.txns.index:
            return None
        row = self.txns.loc[txn_id]
        return _clean({k: row[k] for k in TXN_PUBLIC_FIELDS})

    def decision(self, txn_id: str) -> dict | None:
        if self.decisions is None or txn_id not in self.decisions.index:
            return None
        row = self.decisions.loc[txn_id]
        reasons = row["reasons"]
        if isinstance(reasons, str):
            reasons = json.loads(reasons)
        return {
            "score": float(row["score"]),
            "flagged": bool(row["flagged"]),
            "threshold": float(row["threshold"]),
            "model_version": str(row["model_version"]),
            "reasons": reasons,
        }

    def user_profile(self, user_id: str) -> dict | None:
        if user_id not in self.users.index:
            return None
        u = self.users.loc[user_id]
        return _clean(
            {
                "user_id": user_id,
                "home_city": u["home_city"],
                "account_created_ts": float(u["account_created_ts"]),
            }
        )

    def user_transactions(
        self, user_id: str, before_ts: float, hours: float, limit: int
    ) -> list[dict]:
        if user_id not in self._by_user.groups:
            return []
        g = self._by_user.get_group(user_id)
        g = g[(g.ts < before_ts) & (g.ts >= before_ts - hours * 3600)].tail(limit)
        out = []
        for row in g.itertuples(index=False):
            d = {k: getattr(row, k) for k in TXN_PUBLIC_FIELDS}
            if self.decisions is not None and row.txn_id in self.decisions.index:
                d["model_score"] = float(self.decisions.loc[row.txn_id, "score"])
            out.append(_clean(d))
        return out

    def merchant_profile(self, merchant_id: str, before_ts: float) -> dict | None:
        if merchant_id not in self.merchants.index:
            return None
        m = self.merchants.loc[merchant_id]
        out = {
            "merchant_id": merchant_id,
            "category": m["category"],
            "city": m["city"],
            "country": m["country"],
            "is_online": bool(m["is_online"]),
            "created_ts": float(m["created_ts"]),
            "category_risk_prior": float(m["category_risk"]),
        }
        if merchant_id in self._by_merchant.groups:
            g = self._by_merchant.get_group(merchant_id)
            g = g[g.ts < before_ts]
            out.update(
                {
                    "resolved_txns": int(len(g)),
                    "distinct_users": int(g.user_id.nunique()),
                    "confirmed_fraud_rate": float(g.is_fraud.mean()) if len(g) else None,
                    "median_amount": float(g.amount.median()) if len(g) else None,
                }
            )
        return _clean(out)

    def device_history(self, device_id: str, before_ts: float) -> dict | None:
        if device_id not in self._by_device.groups:
            return {
                "device_id": device_id,
                "first_seen_ts": None,
                "txn_count": 0,
                "distinct_users": 0,
            }
        g = self._by_device.get_group(device_id)
        g = g[g.ts < before_ts]
        resolved = g[g.ts < self.label_cutoff_ts]
        return _clean(
            {
                "device_id": device_id,
                "first_seen_ts": float(g.ts.min()) if len(g) else None,
                "txn_count": int(len(g)),
                "distinct_users": int(g.user_id.nunique()),
                "confirmed_fraud_txns": int(resolved.is_fraud.sum()),
            }
        )

    def similar_cases(self, txn_id: str, k: int) -> list[dict]:
        if txn_id not in self.features.index:
            return []
        x = (self.features.loc[txn_id, FEATURE_NAMES].to_numpy(dtype=float) - self._mu) / self._sd
        x = x / (np.linalg.norm(x) + 1e-9)
        sims = self._bank_x @ x
        idx = np.argsort(-sims)[:k]
        out = []
        for i in idx:
            r = self.bank.iloc[i]
            out.append(
                _clean(
                    {
                        "case_txn_id": r["txn_id"],
                        "similarity": float(sims[i]),
                        "outcome": "fraud" if bool(r["is_fraud"]) else "legit",
                        "fraud_type": str(r["fraud_type"]),
                        "amount": float(r["amount"]),
                        "txn_count_1h": float(r["txn_count_1h"]),
                        "is_new_device": bool(r["is_new_device"]),
                        "speed_kmh": float(r["speed_kmh"]),
                    }
                )
            )
        return out


class DynamoCaseDataSource:
    """Production source backed by the decisions table (recent transactions + scores),
    the reference table (profiles, merchants) and the model's case bank on S3."""

    def __init__(
        self,
        dynamodb,
        decisions_table: str,
        reference_table: str,
        case_bank: pd.DataFrame,
        feature_stats: dict,
    ) -> None:
        self.decisions = dynamodb.Table(decisions_table)
        self.reference = dynamodb.Table(reference_table)
        self.bank = case_bank.reset_index(drop=True)
        self._mu = np.array([feature_stats[f]["mean"] for f in FEATURE_NAMES])
        self._sd = np.array([feature_stats[f]["std"] or 1.0 for f in FEATURE_NAMES])
        bank_x = (self.bank[FEATURE_NAMES].to_numpy() - self._mu) / self._sd
        self._bank_x = bank_x / (np.linalg.norm(bank_x, axis=1, keepdims=True) + 1e-9)

    def _decision_item(self, txn_id: str) -> dict | None:
        return self.decisions.get_item(Key={"txn_id": txn_id}).get("Item")

    def transaction(self, txn_id: str) -> dict | None:
        item = self._decision_item(txn_id)
        if not item:
            return None
        txn = json.loads(item["txn"]) if isinstance(item["txn"], str) else item["txn"]
        return {k: txn.get(k) for k in TXN_PUBLIC_FIELDS}

    def decision(self, txn_id: str) -> dict | None:
        item = self._decision_item(txn_id)
        if not item:
            return None
        return {
            "score": float(item["score"]),
            "flagged": bool(item["flagged"]),
            "threshold": float(item["threshold"]),
            "model_version": item["model_version"],
            "reasons": json.loads(json.dumps(item.get("reasons", []), default=float)),
        }

    def user_profile(self, user_id: str) -> dict | None:
        item = self.reference.get_item(Key={"pk": f"PROFILE#{user_id}"}).get("Item")
        if not item:
            return None
        doc = json.loads(item["doc"])
        return {
            "user_id": user_id,
            "home_city": doc["home_city"],
            "account_created_ts": doc["account_created_ts"],
        }

    def user_transactions(
        self, user_id: str, before_ts: float, hours: float, limit: int
    ) -> list[dict]:
        from boto3.dynamodb.conditions import Key

        resp = self.decisions.query(
            IndexName="by_user",
            KeyConditionExpression=Key("user_id").eq(user_id)
            & Key("ts").between(int(before_ts - hours * 3600), int(before_ts) - 1),
            ScanIndexForward=False,
            Limit=limit,
        )
        out = []
        for item in reversed(resp.get("Items", [])):
            txn = json.loads(item["txn"]) if isinstance(item["txn"], str) else item["txn"]
            d = {k: txn.get(k) for k in TXN_PUBLIC_FIELDS}
            d["model_score"] = float(item["score"])
            out.append(d)
        return out

    def merchant_profile(self, merchant_id: str, before_ts: float) -> dict | None:
        item = self.reference.get_item(Key={"pk": f"MERCHANT#{merchant_id}"}).get("Item")
        if not item:
            return None
        doc = json.loads(item["doc"])
        return {
            "merchant_id": merchant_id,
            "category": doc["category"],
            "city": doc["city"],
            "country": doc.get("country", "IN"),
            "is_online": doc["is_online"],
            "created_ts": doc["created_ts"],
            "category_risk_prior": doc["category_risk"],
        }

    def device_history(self, device_id: str, before_ts: float) -> dict | None:
        item = self.reference.get_item(Key={"pk": f"DEVICE#{device_id}"}).get("Item")
        if not item:
            return {
                "device_id": device_id,
                "first_seen_ts": None,
                "txn_count": 0,
                "distinct_users": 0,
            }
        state = json.loads(item["state"])
        return {
            "device_id": device_id,
            "first_seen_ts": state.get("first_seen_ts"),
            "txn_count": state.get("txn_count", 0),
            "distinct_users": len(state.get("users", [])),
        }

    def similar_cases(self, txn_id: str, k: int) -> list[dict]:
        item = self._decision_item(txn_id)
        if not item:
            return []
        feats = (
            json.loads(item["features"]) if isinstance(item["features"], str) else item["features"]
        )
        x = (np.array([float(feats.get(f, 0.0)) for f in FEATURE_NAMES]) - self._mu) / self._sd
        x = x / (np.linalg.norm(x) + 1e-9)
        sims = self._bank_x @ x
        idx = np.argsort(-sims)[:k]
        return [
            {
                "case_txn_id": str(self.bank.iloc[i]["txn_id"]),
                "similarity": round(float(sims[i]), 4),
                "outcome": "fraud" if bool(self.bank.iloc[i]["is_fraud"]) else "legit",
                "fraud_type": str(self.bank.iloc[i]["fraud_type"]),
            }
            for i in idx
        ]


def load_feature_stats(model_dir: str | Path) -> dict[str, Any]:
    return json.loads((Path(model_dir) / "metadata.json").read_text())["feature_stats"]
