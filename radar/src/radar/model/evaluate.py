"""Evaluation metrics that matter for fraud.

Accuracy is meaningless at a 1% base rate. The operating point that matters to a fraud
team is "how much fraud do we catch if we are only allowed to bother 1 in 100 legitimate
customers" - i.e. recall at a fixed false-positive rate - and how many alerts per day
that produces for analysts.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


def threshold_at_fpr(y_true: np.ndarray, scores: np.ndarray, target_fpr: float) -> float:
    """Smallest threshold whose FPR on (y_true, scores) is <= target_fpr."""
    neg = np.sort(scores[y_true == 0])
    if len(neg) == 0:
        return 0.5
    # the top `target_fpr` fraction of negatives are allowed to be flagged
    k = int(np.floor(target_fpr * len(neg)))
    if k <= 0:
        return float(np.nextafter(neg[-1], np.inf))
    return float(neg[-k])


def metrics_at_threshold(y_true: np.ndarray, scores: np.ndarray, thr: float) -> dict[str, float]:
    pred = scores >= thr
    tp = int((pred & (y_true == 1)).sum())
    fp = int((pred & (y_true == 0)).sum())
    fn = int((~pred & (y_true == 1)).sum())
    tn = int((~pred & (y_true == 0)).sum())
    return {
        "threshold": float(thr),
        "recall": tp / max(1, tp + fn),
        "precision": tp / max(1, tp + fp),
        "fpr": fp / max(1, fp + tn),
        "alerts": tp + fp,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def recall_at_fpr(y_true: np.ndarray, scores: np.ndarray, target_fpr: float) -> float:
    thr = threshold_at_fpr(y_true, scores, target_fpr)
    return metrics_at_threshold(y_true, scores, thr)["recall"]


def summary(y_true: np.ndarray, scores: np.ndarray, thr: float, days: float) -> dict:
    """Threshold-free metrics plus the operating point actually deployed."""
    out = {
        "roc_auc": float(roc_auc_score(y_true, scores)) if len(set(y_true)) > 1 else float("nan"),
        "pr_auc": float(average_precision_score(y_true, scores))
        if len(set(y_true)) > 1
        else float("nan"),
        "n": int(len(y_true)),
        "positives": int(y_true.sum()),
        "base_rate": float(y_true.mean()),
    }
    for fpr in (0.005, 0.01, 0.02):
        out[f"recall_at_{fpr:.3%}_fpr"] = recall_at_fpr(y_true, scores, fpr)
    op = metrics_at_threshold(y_true, scores, thr)
    op["alerts_per_day"] = op["alerts"] / max(days, 1e-9)
    out["operating_point"] = op
    return out


def per_type_recall(df: pd.DataFrame, scores: np.ndarray, thr: float) -> dict[str, dict]:
    """Recall at the deployed threshold broken down by injected fraud pattern."""
    out = {}
    flagged = scores >= thr
    fraud = df["is_fraud"].to_numpy().astype(bool)
    for ftype, grp in df[fraud].groupby("fraud_type"):
        idx = grp.index.to_numpy()
        out[str(ftype)] = {"n": int(len(idx)), "recall": float(flagged[idx].mean())}
    return out


def rule_baseline(X: pd.DataFrame) -> np.ndarray:
    """What a fraud team would ship on day one without ML: a handful of hand-written rules.

    The score is the number of rules that fire, so thresholds are integers and the
    curve is coarse - which is precisely the limitation ML fixes.
    """
    rules = [
        X["amount_z"] > 3,
        X["txn_count_1h"] >= 4,
        (X["is_new_device"] == 1) & (X["amount_ratio_mean"] > 2),
        X["speed_kmh"] > 500,
        X["small_txn_count_1h"] >= 3,
        (X["is_night"] == 1) & (X["category_risk"] >= 0.18),
        X["is_international"] == 1,
        X["device_user_count"] >= 3,
    ]
    return np.sum([r.to_numpy().astype(float) for r in rules], axis=0)
