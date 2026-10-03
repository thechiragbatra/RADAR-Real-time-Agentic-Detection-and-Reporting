import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from radar.agent.investigator import Investigator
from radar.agent.llm import HeuristicBackend, LLMResponse, ScriptedBackend, tool_use_message
from radar.agent.local import build_local_source
from radar.agent.report import render_markdown
from radar.agent.tools import ToolBox, ToolError
from radar.schemas import InvestigationResult, Verdict


@pytest.fixture(scope="module")
def source(data_dir: Path, model_dir: Path):
    src, decisions = build_local_source(data_dir, model_dir)
    return src, decisions


@pytest.fixture(scope="module")
def flagged_txn(source) -> str:
    _, decisions = source
    return decisions[decisions.flagged].iloc[0].txn_id


def _verdict(**over) -> dict:
    base = {
        "verdict": "fraud",
        "confidence": 0.9,
        "fraud_type": "account_takeover",
        "key_evidence": ["device shared by 3 accounts", "amount 6x median"],
        "recommended_action": "block_card_and_contact_customer",
        "analyst_summary": "New shared device, amount far above the customer's normal spend, at night. Block and contact.",
    }
    return {**base, **over}


def test_tools_only_look_backwards(source, flagged_txn):
    src, _ = source
    tb = ToolBox(src, flagged_txn)
    txn = tb.call("get_transaction", {"txn_id": flagged_txn})
    assert "model" in txn and txn["model"]["score"] >= txn["model"]["threshold"]
    hist = tb.call("get_user_transactions", {"user_id": txn["user_id"], "hours": 336, "limit": 100})
    assert all(h["minutes_before_flagged_txn"] > 0 for h in hist["transactions"])
    assert hist["summary"]["count"] == len(hist["transactions"])
    sim = tb.call("find_similar_cases", {"k": 5})
    assert len(sim["cases"]) == 5 and all(c["case_txn_id"] != flagged_txn for c in sim["cases"])
    assert all(
        src.txns.loc[c["case_txn_id"]].ts < src.label_cutoff_ts for c in sim["cases"]
    )  # resolved cases only
    dev = tb.call("get_device_history", {"device_id": txn["device_id"]})
    assert {"distinct_users", "device_age_days"} <= set(dev)


def test_tool_argument_validation(source, flagged_txn):
    src, _ = source
    tb = ToolBox(src, flagged_txn)
    with pytest.raises(ToolError):
        tb.call("get_user_transactions", {"user_id": "U1", "hours": 99999})
    with pytest.raises(ToolError):
        tb.call("get_transaction", {})
    with pytest.raises(ToolError):
        tb.call("drop_tables", {})
    with pytest.raises(ToolError):
        ToolBox(src, "T_DOES_NOT_EXIST")


def test_heuristic_backend_produces_valid_trace(source, flagged_txn):
    src, _ = source
    trace = Investigator(HeuristicBackend(), src).run(flagged_txn)
    assert trace.terminated_by == "submit_verdict"
    assert trace.result.verdict in set(Verdict)
    assert [c.name for c in trace.tool_calls][:1] == ["get_transaction"]
    assert trace.tool_calls[-1].name == "submit_verdict" and all(c.ok for c in trace.tool_calls)
    md = render_markdown(trace)
    assert "Verdict:" in md and flagged_txn in md


def test_unknown_tool_is_refused_and_loop_continues(source, flagged_txn):
    src, _ = source
    backend = ScriptedBackend(
        [
            LLMResponse(tool_use_message("delete_account", {"user_id": "U1"}), "tool_use"),
            LLMResponse(tool_use_message("get_transaction", {"txn_id": flagged_txn}), "tool_use"),
            LLMResponse(tool_use_message("submit_verdict", _verdict()), "tool_use"),
        ]
    )
    trace = Investigator(backend, src).run(flagged_txn)
    assert trace.terminated_by == "submit_verdict"
    assert trace.tool_calls[0].ok is False and "not available" in trace.tool_calls[0].result_preview
    # the refusal went back to the model as an error toolResult
    err = backend.calls[1][-1]["content"][0]["toolResult"]
    assert err["status"] == "error"


def test_invalid_verdict_is_bounced_then_accepted(source, flagged_txn):
    src, _ = source
    bad = _verdict(confidence=1.7, verdict="definitely fraud")
    backend = ScriptedBackend(
        [
            LLMResponse(tool_use_message("get_transaction", {"txn_id": flagged_txn}), "tool_use"),
            LLMResponse(tool_use_message("submit_verdict", bad), "tool_use"),
            LLMResponse(
                tool_use_message(
                    "submit_verdict",
                    _verdict(
                        verdict="legit",
                        confidence=0.7,
                        fraud_type="none",
                        recommended_action="approve",
                    ),
                ),
                "tool_use",
            ),
        ]
    )
    trace = Investigator(backend, src).run(flagged_txn)
    assert trace.result.verdict == Verdict.LEGIT and trace.terminated_by == "submit_verdict"
    assert [c.ok for c in trace.tool_calls] == [True, False, True]


def test_step_budget_yields_needs_review(source, flagged_txn):
    src, _ = source
    backend = ScriptedBackend(
        [LLMResponse(tool_use_message("get_user_profile", {"user_id": "U000001"}), "tool_use")] * 20
    )
    trace = Investigator(backend, src, max_steps=4).run(flagged_txn)
    assert trace.terminated_by == "max_steps"
    assert trace.result.verdict == Verdict.NEEDS_REVIEW and trace.result.confidence == 0.0
    assert trace.steps == 4


def test_prose_ending_is_nudged_to_structured_output(source, flagged_txn):
    src, _ = source
    backend = ScriptedBackend(
        [
            LLMResponse(
                {"role": "assistant", "content": [{"text": "Looks like fraud to me."}]}, "end_turn"
            ),
            LLMResponse(tool_use_message("submit_verdict", _verdict()), "tool_use"),
        ]
    )
    trace = Investigator(backend, src).run(flagged_txn)
    assert trace.result.verdict == Verdict.FRAUD
    nudge = backend.calls[1][-1]
    assert nudge["role"] == "user" and "submit_verdict" in nudge["content"][0]["text"]


def test_token_accounting_and_cost(source, flagged_txn):
    src, _ = source
    backend = ScriptedBackend(
        [
            LLMResponse(
                tool_use_message("get_transaction", {"txn_id": flagged_txn}),
                "tool_use",
                input_tokens=1000,
                output_tokens=100,
            ),
            LLMResponse(
                tool_use_message("submit_verdict", _verdict()),
                "tool_use",
                input_tokens=2000,
                output_tokens=300,
            ),
        ]
    )
    trace = Investigator(backend, src, price_in=3.0, price_out=15.0).run(flagged_txn)
    assert trace.input_tokens == 3000 and trace.output_tokens == 400
    assert trace.cost_usd == pytest.approx((3000 * 3 + 400 * 15) / 1e6)
    assert json.loads(json.dumps(trace.model_dump(mode="json")))["txn_id"] == flagged_txn


def test_submit_verdict_schema_is_strict():
    with pytest.raises(ValidationError):
        InvestigationResult.model_validate(_verdict(key_evidence=[]))
    with pytest.raises(ValidationError):
        InvestigationResult.model_validate(_verdict(analyst_summary="too short"))
