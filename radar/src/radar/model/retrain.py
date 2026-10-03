"""Champion / challenger retraining.

    radar-retrain --data data --registry s3://bucket/models --promote-if-better

Runs as an ECS Fargate task (training can exceed Lambda's 15-minute / 10GB limits),
triggered by the EventBridge ``DriftDetected`` rule or on a weekly schedule.

The challenger is trained on the most recent window and compared with the current
champion on the *same* untouched test window, at the *same* target FPR. It is promoted
only if its recall improves by at least ``--min-improvement`` (absolute) and PR-AUC does
not fall - a one-sided test, because a silent regression costs more than a missed gain.
Every decision, promoted or not, is written to the registry for audit.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import xgboost as xgb

from radar.config import settings
from radar.features import FEATURE_NAMES
from radar.model.evaluate import summary, threshold_at_fpr
from radar.model.registry import ModelRegistry
from radar.model.train import load_features, time_split, train_and_package


def evaluate_champion(
    registry: ModelRegistry,
    data_dir: Path,
    burn_in: float,
    val_days: float,
    test_days: float,
    target_fpr: float,
    workdir: Path,
) -> dict | None:
    champ = registry.champion()
    if champ is None:
        return None
    local = registry.fetch(champ["version"], workdir)
    booster = xgb.Booster()
    booster.load_model(str(local / "model.json"))
    feats = load_features(data_dir)
    _, val, test, _ = time_split(feats, burn_in, val_days, test_days)
    s_val = booster.predict(xgb.DMatrix(val[FEATURE_NAMES], feature_names=FEATURE_NAMES))
    s_test = booster.predict(xgb.DMatrix(test[FEATURE_NAMES], feature_names=FEATURE_NAMES))
    thr = threshold_at_fpr(val["is_fraud"].to_numpy().astype(int), s_val, target_fpr)
    days = (test.ts.max() - test.ts.min()) / 86400
    out = summary(test["is_fraud"].to_numpy().astype(int), np.asarray(s_test), thr, days)
    out["version"] = champ["version"]
    return out


def materialise_data(data_dir: Path) -> None:
    """In AWS the task starts empty: pull the labelled dataset from S3 (RADAR_DATA_S3_PREFIX).

    In production this is where the join between the feature log and the label feed
    (chargebacks, analyst verdicts from the cases table) would happen; the demo retrains
    on the simulator's labelled file.
    """
    if not settings.data_s3_prefix or (data_dir / "transactions.parquet").exists():
        return
    import boto3

    bucket, _, prefix = settings.data_s3_prefix[len("s3://") :].partition("/")
    s3 = boto3.client("s3", region_name=settings.aws_region)
    data_dir.mkdir(parents=True, exist_ok=True)
    for name in ("transactions.parquet", "users.parquet", "merchants.parquet"):
        s3.download_file(bucket, f"{prefix}/{name}", str(data_dir / name))
        print(f"downloaded {name}")


def decide(champion: dict | None, challenger: dict, min_improvement: float) -> tuple[bool, str]:
    key = "recall_at_1.000%_fpr"
    if champion is None:
        return True, "no champion yet"
    gain = challenger[key] - champion[key]
    if gain < min_improvement:
        return False, f"recall@1%FPR gain {gain:+.3f} below required {min_improvement:+.3f}"
    if challenger["pr_auc"] < champion["pr_auc"] - 0.005:
        return False, f"PR-AUC regressed {challenger['pr_auc']:.3f} < {champion['pr_auc']:.3f}"
    return (
        True,
        f"recall@1%FPR {champion[key]:.3f} -> {challenger[key]:.3f} (+{gain:.3f}), PR-AUC {challenger['pr_auc']:.3f}",
    )


def main() -> None:
    p = argparse.ArgumentParser(
        description="Train a challenger and promote it if it beats the champion"
    )
    p.add_argument("--data", type=Path, default=Path(settings.data_dir))
    p.add_argument("--registry", default=settings.model_uri)
    p.add_argument("--workdir", type=Path, default=Path("models"))
    p.add_argument("--burn-in-days", type=float, default=14)
    p.add_argument("--val-days", type=float, default=10)
    p.add_argument("--test-days", type=float, default=14)
    p.add_argument("--target-fpr", type=float, default=0.01)
    p.add_argument("--min-improvement", type=float, default=settings.challenger_min_improvement)
    p.add_argument("--promote-if-better", action="store_true")
    p.add_argument("--reason", default="scheduled retrain")
    a = p.parse_args()

    materialise_data(a.data)
    registry = ModelRegistry(a.registry, settings.aws_region)
    champion_metrics = evaluate_champion(
        registry,
        a.data,
        a.burn_in_days,
        a.val_days,
        a.test_days,
        a.target_fpr,
        a.workdir / "_champion",
    )
    vdir, meta = train_and_package(
        a.data, a.workdir, a.burn_in_days, a.val_days, a.test_days, a.target_fpr
    )
    challenger_metrics = meta["metrics"]["xgboost"]
    registry.publish(vdir, meta["version"])

    promote, why = decide(champion_metrics, challenger_metrics, a.min_improvement)
    decision = {
        "at": time.time(),
        "trigger": a.reason,
        "champion": champion_metrics,
        "challenger": {
            **{k: v for k, v in challenger_metrics.items() if k != "per_fraud_type_recall"},
            "version": meta["version"],
        },
        "promote": promote and a.promote_if_better,
        "reason": why,
    }
    registry.write_text(
        json.dumps(decision, indent=2, default=float), "decisions", f"{meta['version']}.json"
    )
    if promote and a.promote_if_better:
        registry.promote(
            meta["version"],
            f"{a.reason}: {why}",
            {"recall_at_1pct_fpr": challenger_metrics["recall_at_1.000%_fpr"]},
        )
        print(f"PROMOTED {meta['version']}: {why}")
    else:
        print(f"kept champion {champion_metrics['version'] if champion_metrics else None}: {why}")
    print(json.dumps({k: decision[k] for k in ("promote", "reason")}))


if __name__ == "__main__":
    main()
