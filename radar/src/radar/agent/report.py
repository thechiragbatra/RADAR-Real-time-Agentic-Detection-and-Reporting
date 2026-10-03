"""Render an investigation trace as the analyst-facing report (Markdown)."""

from __future__ import annotations

from radar.schemas import InvestigationTrace

ACTION_LABEL = {
    "block_card_and_contact_customer": "Block instrument and contact customer",
    "step_up_authentication": "Step-up authentication on next attempt",
    "monitor": "Monitor account",
    "approve": "Approve / release hold",
}


def render_markdown(trace: InvestigationTrace) -> str:
    r = trace.result
    lines = [
        f"# Investigation {trace.case_id} - transaction {trace.txn_id}",
        "",
        f"**Verdict:** {r.verdict.value.upper()}  ·  **Confidence:** {r.confidence:.0%}  ·  "
        f"**Pattern:** {r.fraud_type.value}  ·  **Action:** {ACTION_LABEL.get(r.recommended_action.value, r.recommended_action.value)}",
        "",
        "## Summary",
        "",
        r.analyst_summary,
        "",
        "## Key evidence",
        "",
        *[f"- {e}" for e in r.key_evidence],
        "",
        "## Investigation trail",
        "",
        f"{len(trace.tool_calls)} tool calls · {trace.steps} model turns · {trace.latency_ms:.0f} ms · "
        f"{trace.input_tokens + trace.output_tokens} tokens · ${trace.cost_usd:.4f} · {trace.backend} ({trace.model_id})"
        + (
            ""
            if trace.terminated_by == "submit_verdict"
            else f" · **terminated by {trace.terminated_by}**"
        ),
        "",
        "| # | tool | arguments | ok |",
        "|---|---|---|---|",
        *[
            f"| {i + 1} | {c.name} | `{_short(c.arguments)}` | {'yes' if c.ok else 'no - ' + c.result_preview[:60]} |"
            for i, c in enumerate(trace.tool_calls)
        ],
        "",
    ]
    return "\n".join(lines)


def _short(d: dict, n: int = 70) -> str:
    s = ", ".join(f"{k}={v}" for k, v in d.items() if k not in ("analyst_summary", "key_evidence"))
    return s if len(s) <= n else s[: n - 3] + "..."
