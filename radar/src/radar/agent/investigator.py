"""The investigation loop and its guardrails.

Guardrails (each one is tested in tests/test_agent.py):

* **Tool allow-list** - a call to a tool that is not in ``ToolBox.specs()`` is refused and
  the refusal is returned to the model as a tool error; it does not crash the loop.
* **Argument validation** - every tool call is validated against a pydantic schema before
  anything touches the data source.
* **Structured output only** - the investigation ends only through ``submit_verdict``,
  whose payload is validated as ``InvestigationResult``; an invalid payload is bounced back
  to the model with the validation error (up to ``max_verdict_retries`` times).
* **Step and token budgets** - a run that exceeds them is closed with ``needs_review`` and
  ``terminated_by`` set, so a stuck model never silently approves or blocks.
* **Read-only** - tools can only read; actions on the account are a downstream decision
  taken by the Lambda from the structured verdict, never by the model directly.
* **Prompt-injection posture** - tool results are data; the system prompt says so and the
  loop never executes anything found inside a tool result.
"""

from __future__ import annotations

import json
import logging
import time
import uuid

from pydantic import ValidationError

from radar.agent.datasource import CaseDataSource
from radar.agent.llm import LLMBackend
from radar.agent.prompts import PROMPT_VERSION, SYSTEM_PROMPT, case_brief
from radar.agent.tools import ToolBox, ToolError
from radar.config import settings
from radar.schemas import (
    InvestigationResult,
    InvestigationTrace,
    RecommendedAction,
    ToolCallRecord,
    Verdict,
)

log = logging.getLogger("radar.agent")


def _preview(obj, n: int = 160) -> str:
    s = json.dumps(obj, default=str)
    return s if len(s) <= n else s[: n - 3] + "..."


class Investigator:
    def __init__(
        self,
        backend: LLMBackend,
        source: CaseDataSource,
        max_steps: int | None = None,
        max_total_tokens: int = 60_000,
        max_verdict_retries: int = 2,
        price_in: float | None = None,
        price_out: float | None = None,
    ) -> None:
        self.backend = backend
        self.source = source
        self.max_steps = max_steps or settings.agent_max_steps
        self.max_total_tokens = max_total_tokens
        self.max_verdict_retries = max_verdict_retries
        self.price_in = settings.price_input_per_mtok if price_in is None else price_in
        self.price_out = settings.price_output_per_mtok if price_out is None else price_out

    def run(self, txn_id: str) -> InvestigationTrace:
        t0 = time.perf_counter()
        case_id = f"case_{uuid.uuid4().hex[:10]}"
        toolbox = ToolBox(self.source, txn_id)
        messages: list[dict] = [{"role": "user", "content": [{"text": case_brief(txn_id)}]}]
        calls: list[ToolCallRecord] = []
        in_tok = out_tok = 0
        verdict_retries = 0
        result: InvestigationResult | None = None
        terminated_by = "submit_verdict"

        for _step in range(1, self.max_steps + 1):
            resp = self.backend.converse(SYSTEM_PROMPT, messages, toolbox.specs())
            in_tok += resp.input_tokens
            out_tok += resp.output_tokens
            messages.append(resp.message)

            tool_uses = [c["toolUse"] for c in resp.message.get("content", []) if "toolUse" in c]
            if not tool_uses:
                if resp.stop_reason == "end_turn":
                    # model tried to finish with prose; push it back to the structured path
                    messages.append(
                        {
                            "role": "user",
                            "content": [
                                {
                                    "text": "Finish by calling submit_verdict with your structured result."
                                }
                            ],
                        }
                    )
                    continue
                terminated_by = f"stop:{resp.stop_reason}"
                break

            tool_results: list[dict] = []
            for tu in tool_uses:
                name, args, tid = tu["name"], tu.get("input") or {}, tu["toolUseId"]
                ts = time.perf_counter()
                if name == "submit_verdict":
                    try:
                        result = InvestigationResult.model_validate(args)
                        calls.append(
                            ToolCallRecord(
                                name=name,
                                arguments=args,
                                ok=True,
                                duration_ms=0.0,
                                result_preview="accepted",
                            )
                        )
                    except ValidationError as exc:
                        verdict_retries += 1
                        err = f"invalid verdict: {exc.errors()[0]['msg']} at {list(exc.errors()[0]['loc'])}"
                        calls.append(
                            ToolCallRecord(
                                name=name,
                                arguments=args,
                                ok=False,
                                duration_ms=0.0,
                                result_preview=err,
                            )
                        )
                        tool_results.append(
                            {
                                "toolResult": {
                                    "toolUseId": tid,
                                    "content": [{"text": err}],
                                    "status": "error",
                                }
                            }
                        )
                        if verdict_retries > self.max_verdict_retries:
                            terminated_by = "invalid_verdict"
                        continue
                    break
                try:
                    if name not in toolbox.names():
                        raise ToolError(f"tool '{name}' is not available")
                    out = toolbox.call(name, args)
                    calls.append(
                        ToolCallRecord(
                            name=name,
                            arguments=args,
                            ok=True,
                            duration_ms=(time.perf_counter() - ts) * 1000,
                            result_preview=_preview(out),
                        )
                    )
                    tool_results.append(
                        {
                            "toolResult": {
                                "toolUseId": tid,
                                "content": [{"json": out}],
                                "status": "success",
                            }
                        }
                    )
                except ToolError as exc:
                    calls.append(
                        ToolCallRecord(
                            name=name,
                            arguments=args,
                            ok=False,
                            duration_ms=(time.perf_counter() - ts) * 1000,
                            result_preview=str(exc),
                        )
                    )
                    tool_results.append(
                        {
                            "toolResult": {
                                "toolUseId": tid,
                                "content": [{"text": str(exc)}],
                                "status": "error",
                            }
                        }
                    )
                except Exception as exc:  # data-source failure: report, do not crash
                    log.exception("tool %s failed", name)
                    calls.append(
                        ToolCallRecord(
                            name=name,
                            arguments=args,
                            ok=False,
                            duration_ms=(time.perf_counter() - ts) * 1000,
                            result_preview=f"error: {exc}",
                        )
                    )
                    tool_results.append(
                        {
                            "toolResult": {
                                "toolUseId": tid,
                                "content": [{"text": f"tool failed: {exc}"}],
                                "status": "error",
                            }
                        }
                    )

            if result is not None or terminated_by != "submit_verdict":
                break
            if tool_results:
                messages.append({"role": "user", "content": tool_results})
            if in_tok + out_tok > self.max_total_tokens:
                terminated_by = "token_budget"
                break
        else:
            terminated_by = "max_steps"

        if result is None:
            result = InvestigationResult(
                verdict=Verdict.NEEDS_REVIEW,
                confidence=0.0,
                key_evidence=[
                    f"investigation did not complete ({terminated_by}); routed to a human"
                ],
                recommended_action=RecommendedAction.STEP_UP_AUTH,
                analyst_summary=f"The automated investigation of {txn_id} was cut short ({terminated_by}) after "
                f"{len(calls)} tool calls. No automated verdict; a human analyst should review the case.",
            )

        cost = (in_tok * self.price_in + out_tok * self.price_out) / 1_000_000
        return InvestigationTrace(
            case_id=case_id,
            txn_id=txn_id,
            result=result,
            steps=len([m for m in messages if m["role"] == "assistant"]),
            tool_calls=calls,
            input_tokens=in_tok,
            output_tokens=out_tok,
            cost_usd=round(cost, 6),
            latency_ms=round((time.perf_counter() - t0) * 1000, 1),
            backend=self.backend.name,
            model_id=f"{self.backend.model_id} / prompt {PROMPT_VERSION}",
            terminated_by=terminated_by,
        )
