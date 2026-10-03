"""Train, evaluate and package the fraud model.

    radar-train --data data --models models --promote

Design choices worth defending in an interview:

* **Time-based split.** Random splits leak the future (a user's later transactions
  inform their earlier ones through the feature store) and overstate performance.
* **Burn-in.** The first N days are replayed for state only, so training rows never
  see a cold feature store that production would not have either.
* **Imbalance.** ``scale_pos_weight`` rather than SMOTE: synthetic positives in a
  behavioural feature space create fraud that never happens. Early stopping on PR-AUC.
* **Operating point** chosen on the validation window at a target FPR and then
  *reported* on the untouched test window - including what FPR it actually achieved.
* **Baselines** (hand-written rules, logistic regression) so the lift is measurable.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from radar.features import FEATURE_NAMES, LocalReferenceData, build_training_frame
from radar.model.drift import build_reference
from radar.model.evaluate import (
    metrics_at_threshold,
    per_type_recall,
    rule_baseline,
    summary,
    threshold_at_fpr,
)
from radar.model.registry import ModelRegistry

DAY = 86400.0

XGB_PARAMS = {
    "objective": "binary:logistic",
    "eval_metric": "aucpr",
    "tree_method": "hist",
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "reg_lambda": 1.0,
    "seed": 42,
}


def load_features(data_dir: Path, cache: bool = True) -> pd.DataFrame:
    cache_path = data_dir / "features.parquet"
    txns = data_dir / "transactions.parquet"
    if cache and cache_path.exists() and cache_path.stat().st_mtime >= txns.stat().st_mtime:
        return pd.read_parquet(cache_path)
    t0 = time.time()
    df = pd.read_parquet(txns)
    ref = LocalReferenceData.from_dir(data_dir)
    feats = build_training_frame(df, ref)
    print(f"featurised {len(feats):,} rows in {time.time() - t0:.1f}s")
    if cache:
        feats.to_parquet(cache_path, index=False)
    return feats


def time_split(
    feats: pd.DataFrame, burn_in_days: float, val_days: float, test_days: float
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    t_min, t_max = feats["ts"].min(), feats["ts"].max()
    test_start = t_max - test_days * DAY
    val_start = test_start - val_days * DAY
    train_start = t_min + burn_in_days * DAY
    train = feats[(feats.ts >= train_start) & (feats.ts < val_start)]
    val = feats[(feats.ts >= val_start) & (feats.ts < test_start)]
    test = feats[feats.ts >= test_start]
    window = {
        "train": [train_start, val_start],
        "val": [val_start, test_start],
        "test": [test_start, t_max],
        "burn_in_days": burn_in_days,
    }
    return train, val, test, window


def fit_xgb(
    train: pd.DataFrame, val: pd.DataFrame, params: dict | None = None, rounds: int = 2000
) -> xgb.Booster:
    p = {**XGB_PARAMS, **(params or {})}
    y_tr = train["is_fraud"].to_numpy().astype(int)
    pos, neg = y_tr.sum(), len(y_tr) - y_tr.sum()
    p["scale_pos_weight"] = float(np.sqrt(neg / max(1, pos)))  # sqrt: full ratio over-fires
    dtr = xgb.DMatrix(train[FEATURE_NAMES], label=y_tr, feature_names=FEATURE_NAMES)
    dva = xgb.DMatrix(
        val[FEATURE_NAMES], label=val["is_fraud"].astype(int), feature_names=FEATURE_NAMES
    )
    booster = xgb.train(
        p,
        dtr,
        num_boost_round=rounds,
        evals=[(dva, "val")],
        early_stopping_rounds=100,
        verbose_eval=False,
    )
    return booster


def predict(booster: xgb.Booster, X: pd.DataFrame) -> np.ndarray:
    d = xgb.DMatrix(X[FEATURE_NAMES], feature_names=FEATURE_NAMES)
    it = (0, booster.best_iteration + 1) if hasattr(booster, "best_iteration") else None
    return booster.predict(d, iteration_range=it) if it else booster.predict(d)


def shap_summary(booster: xgb.Booster, X: pd.DataFrame, n: int = 5000) -> list[dict]:
    import shap

    sample = X[FEATURE_NAMES].sample(min(n, len(X)), random_state=0)
    explainer = shap.TreeExplainer(booster)
    values = explainer.shap_values(sample)
    mean_abs = np.abs(values).mean(axis=0)
    order = np.argsort(-mean_abs)
    return [{"feature": FEATURE_NAMES[i], "mean_abs_shap": float(mean_abs[i])} for i in order]


def build_case_bank(
    train: pd.DataFrame, val: pd.DataFrame, n_legit: int = 4000, n_fraud: int = 4000
) -> pd.DataFrame:
    """Labelled examples the agent's ``find_similar_cases`` tool searches over."""
    pool = pd.concat([train, val])
    fraud = pool[pool.is_fraud].sample(min(n_fraud, int(pool.is_fraud.sum())), random_state=0)
    legit = pool[~pool.is_fraud].sample(min(n_legit, int((~pool.is_fraud).sum())), random_state=0)
    cols = ["txn_id", "ts", "user_id", "is_fraud", "fraud_type", *FEATURE_NAMES]
    return pd.concat([fraud, legit])[cols].reset_index(drop=True)


def train_and_package(
    data_dir: Path,
    out_dir: Path,
    burn_in_days: float = 14,
    val_days: float = 10,
    test_days: float = 10,
    target_fpr: float = 0.01,
    version: str | None = None,
    use_mlflow: bool = False,
) -> tuple[Path, dict]:
    t0 = time.time()
    feats = load_features(data_dir)
    train, val, test, window = time_split(feats, burn_in_days, val_days, test_days)
    print(
        f"train {len(train):,} ({int(train.is_fraud.sum())} fraud) | "
        f"val {len(val):,} ({int(val.is_fraud.sum())}) | test {len(test):,} ({int(test.is_fraud.sum())})"
    )
    y_val = val["is_fraud"].to_numpy().astype(int)
    y_test = test["is_fraud"].to_numpy().astype(int)
    test_days_actual = (test.ts.max() - test.ts.min()) / DAY

    # ---- baselines ---------------------------------------------------------------
    results: dict[str, dict] = {}
    s_rules_val, s_rules_test = rule_baseline(val), rule_baseline(test)
    thr_rules = threshold_at_fpr(y_val, s_rules_val, target_fpr)
    results["rules"] = summary(y_test, s_rules_test, thr_rules, test_days_actual)

    lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced"))
    lr.fit(train[FEATURE_NAMES], train["is_fraud"].astype(int))
    s_lr_val = lr.predict_proba(val[FEATURE_NAMES])[:, 1]
    s_lr_test = lr.predict_proba(test[FEATURE_NAMES])[:, 1]
    results["logistic_regression"] = summary(
        y_test, s_lr_test, threshold_at_fpr(y_val, s_lr_val, target_fpr), test_days_actual
    )

    # ---- champion candidate --------------------------------------------------
    booster = fit_xgb(train, val)
    s_val, s_test = predict(booster, val), predict(booster, test)
    thr = threshold_at_fpr(y_val, s_val, target_fpr)
    results["xgboost"] = summary(y_test, s_test, thr, test_days_actual)
    results["xgboost"]["validation_operating_point"] = metrics_at_threshold(y_val, s_val, thr)
    results["xgboost"]["best_iteration"] = int(booster.best_iteration)
    results["xgboost"]["per_fraud_type_recall"] = per_type_recall(
        test.reset_index(drop=True), s_test, thr
    )

    # ---- package -----------------------------------------------------------------
    version = version or datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    vdir = out_dir / version
    vdir.mkdir(parents=True, exist_ok=True)
    booster.save_model(vdir / "model.json")
    shap_imp = shap_summary(booster, test)
    (vdir / "shap_summary.json").write_text(json.dumps(shap_imp, indent=2))
    reference = build_reference(train, FEATURE_NAMES)
    # prediction drift: the score distribution on validation (train scores would be overfit)
    reference["__score__"] = build_reference(pd.DataFrame({"__score__": s_val}), ["__score__"])[
        "__score__"
    ]
    (vdir / "reference.json").write_text(json.dumps(reference))
    build_case_bank(train, val).to_parquet(vdir / "case_bank.parquet", index=False)
    metadata = {
        "version": version,
        "trained_at": datetime.now(UTC).isoformat(),
        "features": FEATURE_NAMES,
        "threshold": float(thr),
        "target_fpr": target_fpr,
        "window": window,
        "rows": {"train": int(len(train)), "val": int(len(val)), "test": int(len(test))},
        "params": {**XGB_PARAMS, "scale_pos_weight": "sqrt(neg/pos)"},
        "metrics": results,
        "feature_stats": {
            f: {"mean": float(train[f].mean()), "std": float(train[f].std() or 1.0)}
            for f in FEATURE_NAMES
        },
        "training_seconds": round(time.time() - t0, 1),
    }
    (vdir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    (vdir / "report.md").write_text(render_report(metadata, shap_imp))

    if use_mlflow:
        _log_mlflow(metadata, vdir)
    return vdir, metadata


def render_report(meta: dict, shap_imp: list[dict]) -> str:
    m = meta["metrics"]
    rows = []
    for name in ("rules", "logistic_regression", "xgboost"):
        r = m[name]
        op = r["operating_point"]
        rows.append(
            f"| {name} | {r['roc_auc']:.3f} | {r['pr_auc']:.3f} | {r['recall_at_0.500%_fpr']:.1%} | "
            f"{r['recall_at_1.000%_fpr']:.1%} | {r['recall_at_2.000%_fpr']:.1%} | "
            f"{op['recall']:.1%} @ {op['fpr']:.2%} FPR | {op['precision']:.1%} | {op['alerts_per_day']:.0f} |"
        )
    per_type = m["xgboost"]["per_fraud_type_recall"]
    type_rows = [f"| {k} | {v['n']} | {v['recall']:.1%} |" for k, v in sorted(per_type.items())]
    shap_rows = [f"| {s['feature']} | {s['mean_abs_shap']:.4f} |" for s in shap_imp[:15]]
    return "\n".join(
        [
            f"# Model report - version {meta['version']}",
            "",
            f"Trained {meta['trained_at']}. Rows: {meta['rows']}. Target FPR {meta['target_fpr']:.1%} "
            f"(threshold {meta['threshold']:.4f} chosen on validation, metrics below are on the untouched test window).",
            "",
            "| model | ROC-AUC | PR-AUC | recall@0.5%FPR | recall@1%FPR | recall@2%FPR | deployed op. point | precision | alerts/day |",
            "|---|---|---|---|---|---|---|---|---|",
            *rows,
            "",
            "## Recall by fraud pattern (XGBoost, deployed threshold)",
            "",
            "| pattern | n | recall |",
            "|---|---|---|",
            *type_rows,
            "",
            "## Top features by mean |SHAP|",
            "",
            "| feature | mean abs SHAP |",
            "|---|---|",
            *shap_rows,
            "",
        ]
    )


def _log_mlflow(meta: dict, vdir: Path) -> None:
    try:
        import mlflow
    except ImportError:  # pragma: no cover
        print("mlflow not installed; skipping tracking")
        return
    with mlflow.start_run(run_name=meta["version"]):
        mlflow.log_params({k: v for k, v in meta["params"].items() if not isinstance(v, dict)})
        x = meta["metrics"]["xgboost"]
        mlflow.log_metrics(
            {
                "roc_auc": x["roc_auc"],
                "pr_auc": x["pr_auc"],
                "recall_at_1pct_fpr": x["recall_at_1.000%_fpr"],
                "test_precision": x["operating_point"]["precision"],
            }
        )
        mlflow.log_artifacts(str(vdir))


def main() -> None:
    p = argparse.ArgumentParser(description="Train the fraud model")
    p.add_argument("--data", type=Path, default=Path("data"))
    p.add_argument("--models", type=Path, default=Path("models"))
    p.add_argument("--burn-in-days", type=float, default=14)
    p.add_argument("--val-days", type=float, default=10)
    p.add_argument("--test-days", type=float, default=10)
    p.add_argument("--target-fpr", type=float, default=0.01)
    p.add_argument("--version", default=None)
    p.add_argument("--promote", action="store_true", help="make this version the champion")
    p.add_argument("--registry", default=None, help="registry uri (default: --models dir)")
    p.add_argument("--mlflow", action="store_true")
    a = p.parse_args()

    vdir, meta = train_and_package(
        a.data, a.models, a.burn_in_days, a.val_days, a.test_days, a.target_fpr, a.version, a.mlflow
    )
    print((vdir / "report.md").read_text())
    registry = ModelRegistry(a.registry or str(a.models))
    registry.publish(vdir, meta["version"])
    if a.promote:
        registry.promote(
            meta["version"],
            "initial training" if registry.champion() is None else "manual promotion",
            {"recall_at_1pct_fpr": meta["metrics"]["xgboost"]["recall_at_1.000%_fpr"]},
        )
        print(f"promoted {meta['version']} to champion")


if __name__ == "__main__":
    main()
