"""Population Stability Index (PSI) based feature drift detection.

At training time we freeze, for every feature, decile edges and the share of training
rows in each bin (``build_reference``). In production the drift monitor bins the last
24 hours of live features with the same edges and computes

    PSI = sum_i (p_live_i - p_ref_i) * ln(p_live_i / p_ref_i)

Rules of thumb used in credit-risk practice: < 0.1 stable, 0.1-0.2 watch, > 0.2 shifted.
PSI is distribution-only - it needs no labels, so it can run hourly while fraud labels
arrive weeks later from chargebacks.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

EPS = 1e-4


def build_reference(X: pd.DataFrame, features: list[str], bins: int = 10) -> dict[str, dict]:
    ref: dict[str, dict] = {}
    for f in features:
        v = X[f].to_numpy(dtype=float)
        uniq = np.unique(v)
        if len(uniq) <= bins:  # binary / low-cardinality: one bin per value
            edges = np.concatenate([[-np.inf], (uniq[:-1] + uniq[1:]) / 2, [np.inf]])
        else:
            qs = np.quantile(v, np.linspace(0, 1, bins + 1)[1:-1])
            edges = np.concatenate([[-np.inf], np.unique(qs), [np.inf]])
        counts = np.histogram(v, bins=edges)[0]
        ref[f] = {
            "edges": [float(e) for e in edges],
            "probs": (counts / max(1, counts.sum())).tolist(),
        }
    return ref


def psi_for_feature(reference: dict, values: np.ndarray) -> float:
    edges = np.array(reference["edges"], dtype=float)
    p_ref = np.array(reference["probs"], dtype=float)
    counts = np.histogram(values.astype(float), bins=edges)[0]
    p_live = counts / max(1, counts.sum())
    p_ref = np.clip(p_ref, EPS, None)
    p_live = np.clip(p_live, EPS, None)
    return float(np.sum((p_live - p_ref) * np.log(p_live / p_ref)))


def psi_report(reference: dict[str, dict], X_live: pd.DataFrame) -> dict[str, float]:
    return {
        f: psi_for_feature(ref, X_live[f].to_numpy())
        for f, ref in reference.items()
        if f in X_live.columns
    }


def classify(psi: float) -> str:
    if psi < 0.1:
        return "stable"
    if psi < 0.2:
        return "watch"
    return "shifted"
