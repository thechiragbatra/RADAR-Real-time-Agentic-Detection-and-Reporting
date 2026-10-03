"""Build a fully local case data source: score the held-out window with the champion
model so that the agent (and the evaluation harness) see exactly what the stream
consumer would have written to the decisions table."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from radar.agent.datasource import LocalCaseDataSource
from radar.features import FEATURE_NAMES
from radar.scoring.scorer import ModelBundle


def score_window(
    bundle: ModelBundle, feats: pd.DataFrame, explain_flagged: bool = True
) -> pd.DataFrame:
    scores = bundle.score_batch(feats[FEATURE_NAMES].to_dict("records"))
    flagged = scores >= bundle.threshold
    reasons: list[str] = []
    if explain_flagged:
        rows = feats[FEATURE_NAMES].to_dict("records")
        for i, is_flag in enumerate(flagged):
            reasons.append(
                json.dumps([r.model_dump() for r in bundle.explain(rows[i])]) if is_flag else "[]"
            )
    else:
        reasons = ["[]"] * len(feats)
    return pd.DataFrame(
        {
            "txn_id": feats["txn_id"].to_numpy(),
            "ts": feats["ts"].to_numpy(),
            "user_id": feats["user_id"].to_numpy(),
            "score": np.asarray(scores, dtype=float),
            "flagged": flagged,
            "threshold": bundle.threshold,
            "model_version": bundle.version,
            "reasons": reasons,
        }
    )


def build_local_source(
    data_dir: str | Path, model_dir: str | Path, window: str = "test"
) -> tuple[LocalCaseDataSource, pd.DataFrame]:
    data_dir, model_dir = Path(data_dir), Path(model_dir)
    bundle = ModelBundle(model_dir)
    feats = pd.read_parquet(data_dir / "features.parquet")
    start = bundle.metadata["window"][window][0]
    live = feats[feats.ts >= start].reset_index(drop=True)
    decisions = score_window(bundle, live)
    source = LocalCaseDataSource.from_dirs(
        data_dir, model_dir, label_cutoff_ts=start, decisions=decisions
    )
    return source, decisions
