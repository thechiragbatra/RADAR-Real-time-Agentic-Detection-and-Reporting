"""End-to-end demo on a laptop, no AWS needed.

Replays the dataset through the *same* code the Lambdas run: feature store -> scorer ->
decision -> investigation. State is warmed up on everything before the demo window so the
first scored transaction already has history, exactly as in production.

    python scripts/local_stream_demo.py --last 1500 --investigate 5
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd

from radar.agent.investigator import Investigator
from radar.agent.llm import BedrockBackend, HeuristicBackend
from radar.agent.local import build_local_source
from radar.features import InMemoryFeatureStore, LocalReferenceData, featurize_and_update
from radar.lambdas.common import Deps, LocalScorer
from radar.lambdas.stream_consumer import process_transaction
from radar.schemas import Channel, Transaction


def to_txn(row) -> Transaction:
    d = {k: getattr(row, k) for k in Transaction.model_fields}
    d["channel"] = Channel(d["channel"])
    d["is_online"] = bool(d["is_online"])
    return Transaction.model_construct(**d)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, default=Path("data"))
    p.add_argument("--models", type=Path, default=Path("models"))
    p.add_argument(
        "--last", type=int, default=1500, help="transactions to stream through the scorer"
    )
    p.add_argument("--investigate", type=int, default=5, help="investigate the first N flagged")
    p.add_argument("--backend", choices=["heuristic", "bedrock"], default="heuristic")
    a = p.parse_args()

    df = pd.read_parquet(a.data / "transactions.parquet")
    ref = LocalReferenceData.from_dir(a.data)
    store = InMemoryFeatureStore()
    deps = Deps(store=store, ref=ref, scorer=LocalScorer(str(a.models)))

    warm, demo = df.iloc[: -a.last], df.iloc[-a.last :]
    t0 = time.time()
    print(f"warming feature store on {len(warm):,} transactions ...", end=" ", flush=True)
    for row in warm.itertuples(index=False):
        featurize_and_update(to_txn(row), store, ref)
    print(f"{time.time() - t0:.0f}s")

    print(f"\nstreaming {len(demo):,} transactions through the scoring path\n")
    flagged = []
    lat = []
    t0 = time.time()
    for row in demo.itertuples(index=False):
        t = time.perf_counter()
        d = process_transaction(to_txn(row), deps)
        lat.append((time.perf_counter() - t) * 1000)
        if d["flagged"]:
            flagged.append((d, row))
            why = ", ".join(f"{r['feature']}={r['value']:.3g}" for r in d["reasons"])
            truth = f"FRAUD/{row.fraud_type}" if row.is_fraud else f"legit/{row.context}"
            print(
                f"  FLAG {d['txn_id']} score={d['score']:.3f} Rs{d['amount']:>9,.0f} {row.merchant_category:<15} [{why}]  -> {truth}"
            )
    lat_sorted = sorted(lat)
    print(
        f"\n{len(demo):,} scored in {time.time() - t0:.1f}s | flagged {len(flagged)} "
        f"({len(flagged) / len(demo):.2%}) | end-to-end latency p50 {lat_sorted[len(lat) // 2]:.1f} ms "
        f"p95 {lat_sorted[int(len(lat) * 0.95)]:.1f} ms (feature store + score + SHAP for flagged)"
    )
    frauds = sum(1 for _, r in flagged if r.is_fraud)
    print(
        f"flagged: {frauds} fraud, {len(flagged) - frauds} false positives | fraud in window: {int(demo.is_fraud.sum())}"
    )

    if a.investigate and flagged:
        print(
            f"\ninvestigating the first {a.investigate} flagged transactions with the {a.backend} agent\n"
        )
        champ = a.models / "champion.json"
        import json

        version = json.loads(champ.read_text())["version"]
        source, _ = build_local_source(a.data, a.models / version)
        backend = BedrockBackend() if a.backend == "bedrock" else HeuristicBackend()
        inv = Investigator(backend, source)
        for d, row in flagged[: a.investigate]:
            trace = inv.run(d["txn_id"])
            r = trace.result
            truth = f"FRAUD/{row.fraud_type}" if row.is_fraud else f"legit/{row.context}"
            if r.verdict.value == "needs_review":
                mark = "?  "
            else:
                mark = "ok " if (r.verdict.value == "fraud") == bool(row.is_fraud) else "BAD"
            print(
                f"  [{mark}] {d['txn_id']} verdict={r.verdict.value:<12} conf={r.confidence:.2f} action={r.recommended_action.value:<32} truth={truth}"
            )
            specific = [
                e for e in r.key_evidence if not e.startswith("model score")
            ] or r.key_evidence
            print(f"        {specific[0]}")


if __name__ == "__main__":
    main()
