"""Investigate one flagged transaction from the command line.

radar-investigate --txn T000412345                   # heuristic baseline, local data
radar-investigate --txn T000412345 --backend bedrock # real LLM agent via Amazon Bedrock
radar-investigate --random-flagged 3                 # pick flagged txns from the test window
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from radar.agent.investigator import Investigator
from radar.agent.llm import BedrockBackend, HeuristicBackend
from radar.agent.local import build_local_source
from radar.agent.report import render_markdown
from radar.config import settings


def main() -> None:
    p = argparse.ArgumentParser(description="Run the investigation agent on a flagged transaction")
    p.add_argument("--txn", action="append", default=[], help="transaction id (repeatable)")
    p.add_argument(
        "--random-flagged",
        type=int,
        default=0,
        help="also pick N random flagged txns from the test window",
    )
    p.add_argument("--backend", choices=["heuristic", "bedrock"], default="heuristic")
    p.add_argument("--model-id", default=None)
    p.add_argument("--data", type=Path, default=Path(settings.data_dir))
    p.add_argument("--model-dir", type=Path, default=Path(settings.models_dir) / "v1")
    p.add_argument("--json", action="store_true", help="print the raw trace as JSON")
    a = p.parse_args()

    source, decisions = build_local_source(a.data, a.model_dir)
    txns = list(a.txn)
    if a.random_flagged:
        txns += (
            decisions[decisions.flagged].sample(a.random_flagged, random_state=0)["txn_id"].tolist()
        )
    if not txns:
        p.error("give --txn or --random-flagged")

    backend = BedrockBackend(model_id=a.model_id) if a.backend == "bedrock" else HeuristicBackend()
    inv = Investigator(backend, source)
    for txn_id in txns:
        trace = inv.run(txn_id)
        truth = source.txns.loc[txn_id]
        if a.json:
            print(json.dumps(trace.model_dump(mode="json"), indent=2))
        else:
            print(render_markdown(trace))
            print(
                f"> ground truth: {'FRAUD (' + truth['fraud_type'] + ')' if truth['is_fraud'] else 'legit (' + truth['context'] + ')'}\n"
            )


if __name__ == "__main__":
    main()
