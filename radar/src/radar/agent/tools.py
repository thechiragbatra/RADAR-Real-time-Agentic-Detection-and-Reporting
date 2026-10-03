"""Tools the investigation agent can call.

Each tool has a JSON schema (what the model sees), a pydantic model (what we validate
arguments against before executing) and an implementation over ``CaseDataSource``.
Timestamps are returned as ISO strings plus "minutes before the flagged transaction"
because models reason far better about "11 minutes earlier" than about epoch seconds.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

from pydantic import BaseModel, Field, ValidationError

from radar.agent.datasource import CaseDataSource
from radar.schemas import InvestigationResult

IST = timedelta(hours=5, minutes=30)


def _iso(ts: float | None) -> str | None:
    if ts is None:
        return None
    return (datetime.fromtimestamp(ts, tz=UTC) + IST).strftime("%Y-%m-%d %H:%M IST")


class GetTransactionArgs(BaseModel):
    txn_id: str


class GetUserProfileArgs(BaseModel):
    user_id: str


class GetUserTransactionsArgs(BaseModel):
    user_id: str
    hours: float = Field(default=72, ge=1, le=24 * 60, description="Look-back window in hours")
    limit: int = Field(default=25, ge=1, le=100)


class GetMerchantArgs(BaseModel):
    merchant_id: str


class GetDeviceArgs(BaseModel):
    device_id: str


class FindSimilarArgs(BaseModel):
    k: int = Field(default=5, ge=1, le=20)


TOOL_SPECS: list[dict[str, Any]] = [
    {
        "name": "get_transaction",
        "description": "The flagged transaction, the fraud model's score and the top features that drove it.",
        "schema": {
            "type": "object",
            "properties": {"txn_id": {"type": "string"}},
            "required": ["txn_id"],
        },
    },
    {
        "name": "get_user_profile",
        "description": "Static profile of the account: home city, account age.",
        "schema": {
            "type": "object",
            "properties": {"user_id": {"type": "string"}},
            "required": ["user_id"],
        },
    },
    {
        "name": "get_user_transactions",
        "description": "The account's transactions before the flagged one, most recent last, with their model scores. Use a long window (e.g. 336 hours) to look for travel bookings or a phone purchase that would explain unusual activity.",
        "schema": {
            "type": "object",
            "properties": {
                "user_id": {"type": "string"},
                "hours": {
                    "type": "number",
                    "description": "look-back window in hours (default 72, max 1440)",
                },
                "limit": {"type": "integer", "description": "max rows (default 25, max 100)"},
            },
            "required": ["user_id"],
        },
    },
    {
        "name": "get_merchant_profile",
        "description": "Merchant category, location, age, category risk prior and its confirmed fraud rate on resolved cases.",
        "schema": {
            "type": "object",
            "properties": {"merchant_id": {"type": "string"}},
            "required": ["merchant_id"],
        },
    },
    {
        "name": "get_device_history",
        "description": "How long the device has been seen, how many distinct accounts have used it and confirmed fraud on it. A device shared by several accounts is a fraud-ring signal.",
        "schema": {
            "type": "object",
            "properties": {"device_id": {"type": "string"}},
            "required": ["device_id"],
        },
    },
    {
        "name": "find_similar_cases",
        "description": "Nearest resolved cases in feature space with their outcomes - a quick read on how transactions that looked like this one turned out.",
        "schema": {
            "type": "object",
            "properties": {"k": {"type": "integer", "description": "number of cases (default 5)"}},
        },
    },
    {
        "name": "submit_verdict",
        "description": "Submit the final investigation result. This ends the investigation; call it exactly once when you have enough evidence.",
        "schema": InvestigationResult.model_json_schema(),
    },
]

ARG_MODELS: dict[str, type[BaseModel]] = {
    "get_transaction": GetTransactionArgs,
    "get_user_profile": GetUserProfileArgs,
    "get_user_transactions": GetUserTransactionsArgs,
    "get_merchant_profile": GetMerchantArgs,
    "get_device_history": GetDeviceArgs,
    "find_similar_cases": FindSimilarArgs,
    "submit_verdict": InvestigationResult,
}


class ToolError(Exception):
    pass


class ToolBox:
    """Executes tools for one case. Bound to the flagged transaction so that every
    look-back is relative to its timestamp (no peeking at later events)."""

    def __init__(self, source: CaseDataSource, txn_id: str) -> None:
        self.source = source
        self.txn_id = txn_id
        txn = source.transaction(txn_id)
        if txn is None:
            raise ToolError(f"unknown transaction {txn_id}")
        self.txn = txn
        self.anchor_ts: float = float(txn["ts"])

    @staticmethod
    def specs() -> list[dict[str, Any]]:
        return TOOL_SPECS

    @staticmethod
    def names() -> set[str]:
        return {t["name"] for t in TOOL_SPECS}

    def validate(self, name: str, arguments: dict) -> BaseModel:
        model = ARG_MODELS.get(name)
        if model is None:
            raise ToolError(f"unknown tool '{name}'")
        try:
            return model.model_validate(arguments)
        except ValidationError as exc:
            raise ToolError(
                f"invalid arguments for {name}: {exc.errors()[0]['msg']} at {exc.errors()[0]['loc']}"
            ) from exc

    def call(self, name: str, arguments: dict) -> Any:
        args = self.validate(name, arguments)
        handler = getattr(self, f"_t_{name}")
        return handler(args)

    # ---- implementations ---------------------------------------------------------
    def _rel(self, ts: float | None) -> dict:
        if ts is None:
            return {}
        return {
            "when": _iso(ts),
            "minutes_before_flagged_txn": round((self.anchor_ts - ts) / 60, 1),
        }

    def _t_get_transaction(self, a: GetTransactionArgs) -> dict:
        txn = self.source.transaction(a.txn_id)
        if txn is None:
            raise ToolError(f"unknown transaction {a.txn_id}")
        out = {**txn, "when": _iso(txn["ts"])}
        out.pop("ts", None)
        decision = self.source.decision(a.txn_id)
        if decision:
            out["model"] = {
                "score": round(decision["score"], 4),
                "threshold": round(decision["threshold"], 4),
                "version": decision["model_version"],
                "top_reasons": [
                    {
                        "feature": r["feature"],
                        "value": round(float(r["value"]), 3),
                        "effect": r["direction"],
                    }
                    for r in decision.get("reasons", [])
                ],
            }
        return out

    def _t_get_user_profile(self, a: GetUserProfileArgs) -> dict:
        p = self.source.user_profile(a.user_id)
        if p is None:
            raise ToolError(f"unknown user {a.user_id}")
        age_days = (self.anchor_ts - p["account_created_ts"]) / 86400
        return {
            "user_id": p["user_id"],
            "home_city": p["home_city"],
            "account_age_days": round(age_days, 1),
        }

    def _t_get_user_transactions(self, a: GetUserTransactionsArgs) -> dict:
        rows = self.source.user_transactions(a.user_id, self.anchor_ts, a.hours, a.limit)
        out = []
        for r in rows:
            d = {k: v for k, v in r.items() if k not in ("ts", "user_id")}
            d.update(self._rel(r["ts"]))
            out.append(d)
        amounts = [r["amount"] for r in rows]
        summary = {
            "window_hours": a.hours,
            "count": len(rows),
            "median_amount": round(float(sorted(amounts)[len(amounts) // 2]), 2)
            if amounts
            else None,
            "max_amount": round(max(amounts), 2) if amounts else None,
            "distinct_devices": len({r["device_id"] for r in rows}),
            "distinct_cities": sorted({r["city"] for r in rows}),
            "categories": sorted({r["merchant_category"] for r in rows}),
        }
        return {"summary": summary, "transactions": out}

    def _t_get_merchant_profile(self, a: GetMerchantArgs) -> dict:
        m = self.source.merchant_profile(a.merchant_id, self.anchor_ts)
        if m is None:
            raise ToolError(f"unknown merchant {a.merchant_id}")
        out = dict(m)
        out["merchant_age_days"] = round((self.anchor_ts - out.pop("created_ts")) / 86400, 1)
        return out

    def _t_get_device_history(self, a: GetDeviceArgs) -> dict:
        d = self.source.device_history(a.device_id, self.anchor_ts)
        if d is None:
            raise ToolError(f"unknown device {a.device_id}")
        out = dict(d)
        first = out.pop("first_seen_ts", None)
        out["first_seen"] = _iso(first)
        out["device_age_days"] = round((self.anchor_ts - first) / 86400, 2) if first else 0.0
        return out

    def _t_find_similar_cases(self, a: FindSimilarArgs) -> dict:
        cases = self.source.similar_cases(self.txn_id, a.k)
        n_fraud = sum(c["outcome"] == "fraud" for c in cases)
        return {
            "fraud_share_among_neighbours": round(n_fraud / len(cases), 2) if cases else None,
            "cases": cases,
        }

    def _t_submit_verdict(self, a: InvestigationResult) -> dict:
        return a.model_dump(mode="json")
