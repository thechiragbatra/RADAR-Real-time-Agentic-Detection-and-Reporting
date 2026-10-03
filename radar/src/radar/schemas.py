"""Pydantic models shared across the simulator, feature layer, scorer, Lambdas and agent."""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class Channel(StrEnum):
    UPI = "UPI"
    CARD = "CARD"
    NETBANKING = "NETBANKING"


class FraudType(StrEnum):
    NONE = "none"
    ACCOUNT_TAKEOVER = "account_takeover"
    CARD_TESTING = "card_testing"
    VELOCITY_BURST = "velocity_burst"
    GEO_JUMP = "geo_jump"
    MERCHANT_COLLUSION = "merchant_collusion"
    LOW_AND_SLOW = "low_and_slow"


class Transaction(BaseModel):
    """A single payment event as it arrives on the stream.

    No label fields are present on the wire; ``is_fraud`` / ``fraud_type`` only exist
    in the offline training dataset.
    """

    txn_id: str
    ts: float = Field(description="Unix epoch seconds (UTC)")
    user_id: str
    merchant_id: str
    merchant_category: str
    device_id: str
    channel: Channel
    amount: float = Field(gt=0, description="Amount in INR")
    city: str
    lat: float
    lon: float
    country: str = "IN"
    is_online: bool = False


class LabelledTransaction(Transaction):
    is_fraud: bool = False
    fraud_type: FraudType = FraudType.NONE


class UserProfile(BaseModel):
    user_id: str
    home_city: str
    home_lat: float
    home_lon: float
    account_created_ts: float
    spend_level: float = 1.0


class MerchantProfile(BaseModel):
    merchant_id: str
    category: str
    city: str
    is_online: bool
    created_ts: float
    category_risk: float = Field(ge=0, le=1, description="Static MCC-style risk prior")


class ScoreRequest(BaseModel):
    txn_id: str
    features: dict[str, float]
    explain: bool | None = Field(
        default=None,
        description="Force SHAP reasons on/off. Default: explain only when flagged.",
    )


class Reason(BaseModel):
    feature: str
    value: float
    contribution: float
    direction: str  # "raises" | "lowers"


class ScoreResponse(BaseModel):
    txn_id: str
    score: float
    flagged: bool
    threshold: float
    model_version: str
    reasons: list[Reason] = []
    latency_ms: float


class Verdict(StrEnum):
    FRAUD = "fraud"
    LEGIT = "legit"
    NEEDS_REVIEW = "needs_review"


class RecommendedAction(StrEnum):
    BLOCK_AND_CONTACT = "block_card_and_contact_customer"
    STEP_UP_AUTH = "step_up_authentication"
    MONITOR = "monitor"
    APPROVE = "approve"


class InvestigationResult(BaseModel):
    """Structured output the agent must produce via the ``submit_verdict`` tool."""

    verdict: Verdict
    confidence: float = Field(ge=0, le=1)
    fraud_type: FraudType = FraudType.NONE
    key_evidence: list[str] = Field(min_length=1, max_length=8)
    recommended_action: RecommendedAction
    analyst_summary: str = Field(min_length=20, max_length=1200)


class ToolCallRecord(BaseModel):
    name: str
    arguments: dict[str, Any]
    ok: bool
    duration_ms: float
    result_preview: str = ""


class InvestigationTrace(BaseModel):
    case_id: str
    txn_id: str
    result: InvestigationResult
    steps: int
    tool_calls: list[ToolCallRecord]
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    latency_ms: float = 0.0
    backend: str
    model_id: str
    terminated_by: str = "submit_verdict"  # or "max_steps" | "error"
