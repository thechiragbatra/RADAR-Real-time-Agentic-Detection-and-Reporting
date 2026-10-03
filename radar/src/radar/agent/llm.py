"""LLM backends behind one interface shaped like Bedrock's Converse API.

    BedrockBackend    Amazon Bedrock Converse with tool use (production)
    HeuristicBackend  deterministic rule-based "analyst" that drives the same tools -
                      the baseline the LLM is measured against, and what tests use
    ScriptedBackend   replays canned responses (tests for the loop and guardrails)

Messages use the Converse wire format: ``{"role": ..., "content": [{"text"} | {"toolUse"}
| {"toolResult"}]}``. Keeping that format as the internal lingua franca means the agent
loop has no backend-specific branches.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any, Protocol

from radar.config import settings


@dataclass
class LLMResponse:
    message: dict[str, Any]  # assistant message in Converse format
    stop_reason: str  # "tool_use" | "end_turn" | "max_tokens" | ...
    input_tokens: int = 0
    output_tokens: int = 0


class LLMBackend(Protocol):
    name: str
    model_id: str

    def converse(self, system: str, messages: list[dict], tools: list[dict]) -> LLMResponse: ...


def tool_use_message(name: str, arguments: dict, text: str | None = None) -> dict:
    content: list[dict] = []
    if text:
        content.append({"text": text})
    content.append(
        {
            "toolUse": {
                "toolUseId": f"tooluse_{uuid.uuid4().hex[:12]}",
                "name": name,
                "input": arguments,
            }
        }
    )
    return {"role": "assistant", "content": content}


# ---- Bedrock -----------------------------------------------------------------------------
class BedrockBackend:
    name = "bedrock"

    def __init__(
        self,
        model_id: str | None = None,
        region: str | None = None,
        client=None,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> None:
        self.model_id = model_id or settings.bedrock_model_id
        self.max_tokens = max_tokens or settings.agent_max_tokens
        self.temperature = settings.agent_temperature if temperature is None else temperature
        if client is None:
            import boto3
            from botocore.config import Config

            client = boto3.client(
                "bedrock-runtime",
                region_name=region or settings.bedrock_region,
                config=Config(
                    read_timeout=60,
                    connect_timeout=10,
                    retries={"max_attempts": 3, "mode": "adaptive"},
                ),
            )
        self.client = client

    def converse(self, system: str, messages: list[dict], tools: list[dict]) -> LLMResponse:
        resp = self.client.converse(
            modelId=self.model_id,
            system=[{"text": system}],
            messages=messages,
            toolConfig={
                "tools": [
                    {
                        "toolSpec": {
                            "name": t["name"],
                            "description": t["description"],
                            "inputSchema": {"json": t["schema"]},
                        }
                    }
                    for t in tools
                ]
            },
            inferenceConfig={"maxTokens": self.max_tokens, "temperature": self.temperature},
        )
        usage = resp.get("usage", {})
        return LLMResponse(
            message=resp["output"]["message"],
            stop_reason=resp.get("stopReason", "end_turn"),
            input_tokens=int(usage.get("inputTokens", 0)),
            output_tokens=int(usage.get("outputTokens", 0)),
        )


# ---- Scripted (tests) -------------------------------------------------------------------
class ScriptedBackend:
    name = "scripted"
    model_id = "scripted"

    def __init__(self, responses: list[LLMResponse]) -> None:
        self.responses = list(responses)
        self.calls: list[list[dict]] = []

    def converse(self, system: str, messages: list[dict], tools: list[dict]) -> LLMResponse:
        self.calls.append(list(messages))  # snapshot: the loop mutates the list afterwards
        if not self.responses:
            raise RuntimeError("scripted backend exhausted")
        return self.responses.pop(0)


# ---- Heuristic baseline -------------------------------------------------------------
@dataclass
class _Gathered:
    txn: dict | None = None
    profile: dict | None = None
    history: dict | None = None
    merchant: dict | None = None
    device: dict | None = None
    similar: dict | None = None
    extra: dict = field(default_factory=dict)


class HeuristicBackend:
    """A fixed investigation plan plus hand-written decision rules.

    It is intentionally the kind of thing a fraud-ops team could write in an afternoon.
    If the LLM agent cannot beat it on the labelled evaluation set, the LLM is not
    earning its cost.
    """

    name = "heuristic"
    model_id = "heuristic-v1"
    PLAN = [
        "get_transaction",
        "get_user_profile",
        "get_user_transactions",
        "get_device_history",
        "get_merchant_profile",
        "find_similar_cases",
    ]

    def converse(self, system: str, messages: list[dict], tools: list[dict]) -> LLMResponse:
        results = self._tool_results(messages)
        step = len(results)
        txn_id = self._txn_id_from_brief(messages)
        if step < len(self.PLAN):
            name = self.PLAN[step]
            args = self._args_for(name, txn_id, results)
            return LLMResponse(tool_use_message(name, args), "tool_use")
        verdict = self._decide(results)
        return LLMResponse(tool_use_message("submit_verdict", verdict), "tool_use")

    # ---- helpers --------------------------------------------------------------------------
    @staticmethod
    def _txn_id_from_brief(messages: list[dict]) -> str:
        text = messages[0]["content"][0].get("text", "")
        for tok in text.replace(".", " ").split():
            if tok.startswith("T") and tok[1:].isdigit():
                return tok
        return ""

    @staticmethod
    def _tool_results(messages: list[dict]) -> list[tuple[str, Any]]:
        """(tool name, result json) pairs in order, pairing toolUse ids with toolResults."""
        names: dict[str, str] = {}
        out: list[tuple[str, Any]] = []
        for m in messages:
            for c in m.get("content", []):
                if "toolUse" in c:
                    names[c["toolUse"]["toolUseId"]] = c["toolUse"]["name"]
                if "toolResult" in c:
                    tr = c["toolResult"]
                    body = tr["content"][0]
                    out.append(
                        (names.get(tr["toolUseId"], "?"), body.get("json", body.get("text")))
                    )
        return out

    def _args_for(self, name: str, txn_id: str, results: list[tuple[str, Any]]) -> dict:
        txn = next((r for n, r in results if n == "get_transaction"), None) or {}
        if name == "get_transaction":
            return {"txn_id": txn_id}
        if name == "get_user_profile":
            return {"user_id": txn.get("user_id", "")}
        if name == "get_user_transactions":
            return {"user_id": txn.get("user_id", ""), "hours": 336, "limit": 60}
        if name == "get_device_history":
            return {"device_id": txn.get("device_id", "")}
        if name == "get_merchant_profile":
            return {"merchant_id": txn.get("merchant_id", "")}
        return {"k": 7}

    def _decide(self, results: list[tuple[str, Any]]) -> dict:
        g = _Gathered()
        for n, r in results:
            if not isinstance(r, dict):
                continue
            if n == "get_transaction":
                g.txn = r
            elif n == "get_user_profile":
                g.profile = r
            elif n == "get_user_transactions":
                g.history = r
            elif n == "get_device_history":
                g.device = r
            elif n == "get_merchant_profile":
                g.merchant = r
            elif n == "find_similar_cases":
                g.similar = r

        pts = 0.0
        evidence: list[str] = []
        txn = g.txn or {}
        model = txn.get("model", {})
        score = float(model.get("score", 0.0))
        reasons = {r["feature"]: r for r in model.get("top_reasons", [])}
        hist = (g.history or {}).get("transactions", [])
        summ = (g.history or {}).get("summary", {})
        amount = float(txn.get("amount", 0.0))
        median = summ.get("median_amount") or amount
        device = g.device or {}
        merchant = g.merchant or {}
        similar = g.similar or {}

        if score >= 0.6:
            pts += 2
            evidence.append(
                f"model score {score:.2f} is far above threshold {model.get('threshold', 0):.2f}"
            )
        elif score >= float(model.get("threshold", 0.5)):
            pts += 1
            evidence.append(
                f"model score {score:.2f} above threshold {model.get('threshold', 0):.2f}"
            )

        dev_users = int(device.get("distinct_users", 0))
        dev_age = float(device.get("device_age_days", 0.0))
        new_to_account = txn.get("device_id") not in {h.get("device_id") for h in hist}
        if new_to_account and dev_users >= 2:
            pts += 2.5
            evidence.append(
                f"device {txn.get('device_id')} is new to this account and shared by {dev_users} accounts"
            )
        elif new_to_account:
            pts += 1
            evidence.append(
                f"device {txn.get('device_id')} never seen on this account before (device age {dev_age:.1f} days)"
            )
        if device.get("confirmed_fraud_txns", 0) > 0:
            pts += 2
            evidence.append(
                f"device has {device['confirmed_fraud_txns']} confirmed fraud transactions"
            )

        last_hour = [h for h in hist if h.get("minutes_before_flagged_txn", 1e9) <= 60]
        if len(last_hour) >= 4:
            pts += 1.5
            evidence.append(f"{len(last_hour)} transactions in the previous hour")
        small = [h for h in last_hour if h.get("amount", 0) < 100]
        if len(small) >= 3:
            pts += 1.5
            evidence.append(
                f"{len(small)} sub-Rs100 transactions in the previous hour (card testing pattern)"
            )
        if median and amount > 3 * median:
            pts += 1
            evidence.append(
                f"amount Rs{amount:,.0f} is {amount / median:.1f}x the account's 2-week median Rs{median:,.0f}"
            )
        if "speed_kmh" in reasons and reasons["speed_kmh"]["value"] > 500:
            pts += 2
            evidence.append(
                f"implied travel speed {reasons['speed_kmh']['value']:.0f} km/h since previous transaction"
            )
        if txn.get("country", "IN") != "IN":
            pts += 1.5
            evidence.append(f"international transaction in {txn.get('city')}, {txn.get('country')}")
        risk = float(merchant.get("category_risk_prior", 0.0))
        mrate = merchant.get("confirmed_fraud_rate")
        if risk >= 0.2 or (mrate is not None and mrate > 0.05):
            pts += 1
            evidence.append(
                f"merchant category {merchant.get('category')} is high risk (prior {risk:.2f}, confirmed rate {mrate})"
            )
        share = similar.get("fraud_share_among_neighbours")
        if share is not None and share >= 0.6:
            pts += 1
            evidence.append(f"{share:.0%} of the most similar resolved cases were fraud")
        when = txn.get("when", "")
        hour = int(when[11:13]) if len(when) > 13 else 12
        night = hour < 5 or hour >= 23
        if night and not any(
            (int(h.get("when", "x" * 13)[11:13]) < 5) for h in hist if len(h.get("when", "")) > 13
        ):
            pts += 0.5
            evidence.append(
                f"transaction at {when[11:16]} IST; no night activity in the account's history"
            )

        # ---- mitigations ------------------------------------------------------------
        booking = [h for h in hist if h.get("merchant_category") == "travel"]
        home = (g.profile or {}).get("home_city")
        if booking and txn.get("city") not in (home, None) and txn.get("country", "IN") == "IN":
            pts -= 2.5
            evidence.append(
                f"travel booking {booking[-1].get('minutes_before_flagged_txn', 0) / 1440:.0f} days earlier explains purchase in {txn.get('city')}"
            )
        phone = [
            h
            for h in hist
            if h.get("merchant_category") == "electronics"
            and h.get("minutes_before_flagged_txn", 1e9) <= 3 * 1440
        ]
        if phone and new_to_account and dev_users <= 1:
            pts -= 2
            evidence.append(
                "electronics purchase in the last 3 days on the old device: consistent with a phone upgrade"
            )
        if not new_to_account and dev_age > 30:
            pts -= 1
            evidence.append(f"device has been used on this account for {dev_age:.0f} days")
        if txn.get("merchant_id") in {h.get("merchant_id") for h in hist}:
            pts -= 1
            evidence.append("merchant already used by this account in the last 2 weeks")
        if share is not None and share <= 0.2:
            pts -= 1
            evidence.append(f"only {share:.0%} of similar resolved cases were fraud")

        if pts >= 3:
            verdict, conf = "fraud", min(0.95, 0.6 + 0.07 * pts)
        elif pts <= 0.5:
            verdict, conf = "legit", min(0.9, 0.6 + 0.1 * (0.5 - pts))
        else:
            verdict, conf = "needs_review", 0.5

        if len(small) >= 3:
            ftype = "card_testing"
        elif (
            "speed_kmh" in reasons
            and reasons["speed_kmh"]["value"] > 500
            or txn.get("country", "IN") != "IN"
        ):
            ftype = "geo_jump"
        elif new_to_account and dev_users >= 2:
            ftype = "account_takeover"
        elif len(last_hour) >= 4:
            ftype = "velocity_burst"
        elif risk >= 0.2 and (median and amount > 2 * median):
            ftype = "merchant_collusion"
        elif new_to_account:
            ftype = "low_and_slow" if amount <= 2 * (median or amount) else "account_takeover"
        else:
            ftype = "none"
        if verdict != "fraud":
            ftype = "none"

        action = {
            "fraud": "block_card_and_contact_customer" if conf >= 0.8 else "step_up_authentication",
            "needs_review": "step_up_authentication",
            "legit": "approve",
        }[verdict]
        summary = (
            f"Transaction {txn.get('txn_id')} of Rs{amount:,.0f} at a {merchant.get('category', 'unknown')} merchant scored {score:.2f}. "
            f"Evidence points: {pts:+.1f}. Strongest signals: {'; '.join(evidence[:3]) if evidence else 'none beyond the model score'}. "
            f"Verdict {verdict} ({conf:.0%}); recommended action: {action.replace('_', ' ')}."
        )
        return {
            "verdict": verdict,
            "confidence": round(conf, 2),
            "fraud_type": ftype,
            "key_evidence": (evidence or [f"model score {score:.2f}"])[:8],
            "recommended_action": action,
            "analyst_summary": summary[:1200],
        }
