"""Evaluation harness for the investigation agent.

    radar-eval build-cases --n 100                 # pick a stratified, labelled case set
    radar-eval run --backend heuristic             # baseline investigator
    radar-eval run --backend bedrock               # LLM agent
    radar-eval compare                             # side-by-side table of all runs

Cases are flagged transactions from the held-out window: half confirmed fraud (spread
across the injected patterns), half false positives (spread across the legitimate
behaviours that fooled the model). The agent's job is to tell them apart, so the numbers
that matter are:

* **coverage** - share of cases where it committed to fraud/legit instead of abstaining
* **decided accuracy** - accuracy on those it committed to
* **false blocks** - legitimate customers it would have blocked (verdict fraud, conf >= 0.8)
* **missed fraud** - fraud it would have released (verdict legit)
* **cost per case** and **latency**, because an investigator that is right but slow or
  expensive does not get deployed.

The heuristic backend is the bar: an LLM that cannot beat it is not worth its tokens.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from radar.agent.investigator import Investigator
from radar.agent.llm import BedrockBackend, HeuristicBackend
from radar.agent.local import build_local_source
from radar.config import settings
from radar.schemas import InvestigationTrace

CASES_PATH = Path("evals/cases/cases.jsonl")
RESULTS_DIR = Path("evals/results")


# ---- case selection -------------------------------------------------------------------
def build_cases(data_dir: Path, model_dir: Path, n: int, seed: int = 0) -> pd.DataFrame:
    source, decisions = build_local_source(data_dir, model_dir)
    truth = source.txns[["txn_id", "is_fraud", "fraud_type", "context", "amount"]].reset_index(
        drop=True
    )
    flagged = decisions[decisions.flagged].merge(truth, on="txn_id")
    half = n // 2
    fraud = _stratified(flagged[flagged.is_fraud], "fraud_type", half, seed)
    legit = _stratified(flagged[~flagged.is_fraud], "context", n - half, seed)
    cases = pd.concat([fraud, legit]).sample(frac=1, random_state=seed).reset_index(drop=True)
    cases["stratum"] = cases.apply(
        lambda r: r.fraud_type if r.is_fraud else f"legit:{r.context}", axis=1
    )
    return cases[["txn_id", "is_fraud", "fraud_type", "context", "stratum", "score", "amount"]]


def _stratified(df: pd.DataFrame, col: str, n: int, seed: int) -> pd.DataFrame:
    """Round-robin across strata so rare patterns are represented."""
    groups = {k: g.sample(frac=1, random_state=seed) for k, g in df.groupby(col)}
    picked: list[pd.DataFrame] = []
    idx = dict.fromkeys(groups, 0)
    total = 0
    while total < n and any(idx[k] < len(g) for k, g in groups.items()):
        for k, g in groups.items():
            if idx[k] < len(g) and total < n:
                picked.append(g.iloc[[idx[k]]])
                idx[k] += 1
                total += 1
    return pd.concat(picked) if picked else df.head(0)


# ---- scoring a run -------------------------------------------------------------------------
def score_run(cases: pd.DataFrame, traces: list[InvestigationTrace]) -> dict:
    by_id = {t.txn_id: t for t in traces}
    rows = []
    for c in cases.itertuples(index=False):
        t = by_id[c.txn_id]
        v = t.result.verdict.value
        decided = v != "needs_review"
        correct = (v == "fraud") == bool(c.is_fraud) if decided else None
        blocks = v == "fraud" and t.result.confidence >= 0.8
        rows.append(
            {
                "txn_id": c.txn_id,
                "stratum": c.stratum,
                "truth": "fraud" if c.is_fraud else "legit",
                "verdict": v,
                "confidence": t.result.confidence,
                "decided": decided,
                "correct": correct,
                "false_block": blocks and not c.is_fraud,
                "missed_fraud": v == "legit" and bool(c.is_fraud),
                "type_match": bool(c.is_fraud)
                and v == "fraud"
                and t.result.fraud_type.value == c.fraud_type,
                "tool_calls": len(t.tool_calls),
                "failed_tool_calls": sum(not x.ok for x in t.tool_calls),
                "latency_ms": t.latency_ms,
                "tokens": t.input_tokens + t.output_tokens,
                "cost_usd": t.cost_usd,
                "terminated_by": t.terminated_by,
            }
        )
    df = pd.DataFrame(rows)
    decided = df[df.decided]
    fraud_truth = df[df.truth == "fraud"]
    fraud_verdict = df[df.verdict == "fraud"]
    out = {
        "n": int(len(df)),
        "coverage": float(df.decided.mean()),
        "decided_accuracy": float(decided.correct.mean()) if len(decided) else None,
        "fraud_recall": float((fraud_truth.verdict == "fraud").mean())
        if len(fraud_truth)
        else None,
        "fraud_precision": float((fraud_verdict.truth == "fraud").mean())
        if len(fraud_verdict)
        else None,
        "false_blocks": int(df.false_block.sum()),
        "missed_fraud": int(df.missed_fraud.sum()),
        "fraud_type_accuracy": float(
            df[(df.truth == "fraud") & (df.verdict == "fraud")].type_match.mean()
        )
        if len(fraud_verdict)
        else None,
        "incomplete_runs": int((df.terminated_by != "submit_verdict").sum()),
        "avg_tool_calls": float(df.tool_calls.mean()),
        "failed_tool_calls": int(df.failed_tool_calls.sum()),
        "p50_latency_ms": float(statistics.median(df.latency_ms)),
        "p95_latency_ms": float(df.latency_ms.quantile(0.95)),
        "avg_tokens": float(df.tokens.mean()),
        "avg_cost_usd": float(df.cost_usd.mean()),
        "monthly_cost_usd_at_100_alerts_per_day": float(df.cost_usd.mean() * 100 * 30),
        "by_stratum": {
            k: {
                "n": int(len(g)),
                "coverage": float(g.decided.mean()),
                "decided_accuracy": float(g[g.decided].correct.mean()) if g.decided.any() else None,
            }
            for k, g in df.groupby("stratum")
        },
        "confusion": df.groupby(["truth", "verdict"]).size().to_dict(),
        "rows": rows,
    }
    out["confusion"] = {f"{k[0]}->{k[1]}": int(v) for k, v in out["confusion"].items()}
    return out


# ---- reporting ----------------------------------------------------------------------------
def render_compare(results: dict[str, dict]) -> str:
    cols = [
        "coverage",
        "decided_accuracy",
        "fraud_recall",
        "fraud_precision",
        "false_blocks",
        "missed_fraud",
        "fraud_type_accuracy",
        "avg_tool_calls",
        "p50_latency_ms",
        "avg_tokens",
        "avg_cost_usd",
        "monthly_cost_usd_at_100_alerts_per_day",
    ]
    lines = ["| metric | " + " | ".join(results) + " |", "|---|" + "---|" * len(results)]
    for c in cols:
        vals = []
        for r in results.values():
            v = r.get(c)
            if v is None:
                vals.append("-")
            elif c in ("false_blocks", "missed_fraud"):
                vals.append(f"{int(v)}")
            elif c.endswith("_ms"):
                vals.append(f"{v:,.0f}")
            elif c.startswith("avg_cost") or c.startswith("monthly"):
                vals.append(f"${v:,.4f}" if c.startswith("avg") else f"${v:,.0f}")
            elif c in ("avg_tool_calls", "avg_tokens"):
                vals.append(f"{v:,.1f}")
            else:
                vals.append(f"{v:.1%}")
        lines.append(f"| {c} | " + " | ".join(vals) + " |")
    strata = sorted({s for r in results.values() for s in r["by_stratum"]})
    lines += [
        "",
        "### Decided accuracy by stratum",
        "",
        "| stratum | n | " + " | ".join(results) + " |",
        "|---|---|" + "---|" * len(results),
    ]
    for s in strata:
        n = next((r["by_stratum"][s]["n"] for r in results.values() if s in r["by_stratum"]), 0)
        vals = []
        for r in results.values():
            b = r["by_stratum"].get(s)
            vals.append(
                "-"
                if not b or b["decided_accuracy"] is None
                else f"{b['decided_accuracy']:.0%} (cov {b['coverage']:.0%})"
            )
        lines.append(f"| {s} | {n} | " + " | ".join(vals) + " |")
    return "\n".join(lines) + "\n"


# ---- CLI ----------------------------------------------------------------------------------------
def main() -> None:
    p = argparse.ArgumentParser(description="Investigation agent evaluation harness")
    sub = p.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build-cases")
    b.add_argument("--n", type=int, default=100)
    b.add_argument("--seed", type=int, default=0)
    b.add_argument("--data", type=Path, default=Path(settings.data_dir))
    b.add_argument("--model-dir", type=Path, default=Path(settings.models_dir) / "v1")
    b.add_argument("--out", type=Path, default=CASES_PATH)

    r = sub.add_parser("run")
    r.add_argument("--backend", choices=["heuristic", "bedrock"], default="heuristic")
    r.add_argument("--model-id", default=None)
    r.add_argument("--cases", type=Path, default=CASES_PATH)
    r.add_argument("--data", type=Path, default=Path(settings.data_dir))
    r.add_argument("--model-dir", type=Path, default=Path(settings.models_dir) / "v1")
    r.add_argument("--limit", type=int, default=0)
    r.add_argument("--label", default=None, help="name for this run (default: backend)")

    c = sub.add_parser("compare")
    c.add_argument("--results", type=Path, default=RESULTS_DIR)

    a = p.parse_args()
    if a.cmd == "build-cases":
        cases = build_cases(a.data, a.model_dir, a.n, a.seed)
        a.out.parent.mkdir(parents=True, exist_ok=True)
        cases.to_json(a.out, orient="records", lines=True)
        print(f"wrote {len(cases)} cases to {a.out}")
        print(cases.stratum.value_counts().to_string())
    elif a.cmd == "run":
        cases = pd.read_json(a.cases, lines=True)
        if a.limit:
            cases = cases.head(a.limit)
        source, _ = build_local_source(a.data, a.model_dir)
        backend = (
            BedrockBackend(model_id=a.model_id) if a.backend == "bedrock" else HeuristicBackend()
        )
        inv = Investigator(backend, source)
        traces: list[InvestigationTrace] = []
        t0 = time.time()
        for i, txn_id in enumerate(cases.txn_id, 1):
            traces.append(inv.run(txn_id))
            if i % 10 == 0:
                print(f"{i}/{len(cases)} cases, {time.time() - t0:.0f}s")
        result = score_run(cases, traces)
        label = a.label or (
            f"{a.backend}:{backend.model_id.split('/')[0]}" if a.backend == "bedrock" else a.backend
        )
        result.update(
            {
                "label": label,
                "backend": a.backend,
                "model_id": backend.model_id,
                "run_at": datetime.now(UTC).isoformat(),
                "cases_file": str(a.cases),
            }
        )
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        out = RESULTS_DIR / f"{label.replace(':', '_').replace('.', '_')}.json"
        out.write_text(json.dumps(result, indent=2, default=str))
        (RESULTS_DIR / f"{out.stem}.traces.jsonl").write_text(
            "\n".join(json.dumps(t.model_dump(mode="json")) for t in traces)
        )
        print(
            json.dumps(
                {k: v for k, v in result.items() if k not in ("rows", "by_stratum", "confusion")},
                indent=2,
            )
        )
        print("confusion:", result["confusion"])
        print(f"saved {out}")
    else:
        results = {}
        for f in sorted(a.results.glob("*.json")):
            d = json.loads(f.read_text())
            results[d.get("label", f.stem)] = d
        if not results:
            print("no results yet")
            return
        md = render_compare(results)
        (a.results / "comparison.md").write_text(md)
        print(md)


if __name__ == "__main__":
    main()
