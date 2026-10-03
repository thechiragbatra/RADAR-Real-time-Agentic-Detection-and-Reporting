import json
from pathlib import Path

import numpy as np
import pandas as pd

from radar.features import FEATURE_NAMES
from radar.model.drift import build_reference, classify, psi_report
from radar.model.evaluate import (
    metrics_at_threshold,
    recall_at_fpr,
    rule_baseline,
    threshold_at_fpr,
)
from radar.model.registry import ModelRegistry
from radar.model.retrain import decide


def test_threshold_at_fpr_hits_target():
    rng = np.random.default_rng(0)
    y = np.r_[np.zeros(10_000), np.ones(100)]
    s = np.r_[rng.beta(2, 8, 10_000), rng.beta(8, 2, 100)]
    for fpr in (0.005, 0.01, 0.02):
        thr = threshold_at_fpr(y, s, fpr)
        m = metrics_at_threshold(y, s, thr)
        assert m["fpr"] <= fpr + 1e-9 and m["fpr"] > fpr - 0.002
    assert recall_at_fpr(y, s, 0.01) > 0.8


def test_rule_baseline_shape(data_dir):
    feats = pd.read_parquet(data_dir / "features.parquet").head(500)
    s = rule_baseline(feats)
    assert s.shape == (500,) and s.min() >= 0 and s.max() <= 8


def test_psi_detects_shift():
    rng = np.random.default_rng(1)
    ref_df = pd.DataFrame(
        {"a": rng.normal(0, 1, 20_000), "b": rng.integers(0, 2, 20_000).astype(float)}
    )
    ref = build_reference(ref_df, ["a", "b"])
    same = pd.DataFrame(
        {"a": rng.normal(0, 1, 5_000), "b": rng.integers(0, 2, 5_000).astype(float)}
    )
    shifted = pd.DataFrame(
        {"a": rng.normal(1.0, 1.3, 5_000), "b": (rng.random(5_000) < 0.9).astype(float)}
    )
    p_same, p_shift = psi_report(ref, same), psi_report(ref, shifted)
    assert p_same["a"] < 0.05 and p_same["b"] < 0.05
    assert p_shift["a"] > 0.2 and p_shift["b"] > 0.2
    assert classify(p_same["a"]) == "stable" and classify(p_shift["a"]) == "shifted"


def test_trained_bundle_is_complete_and_beats_baselines(model_dir: Path):
    for name in (
        "model.json",
        "metadata.json",
        "reference.json",
        "shap_summary.json",
        "case_bank.parquet",
        "report.md",
    ):
        assert (model_dir / name).exists(), name
    meta = json.loads((model_dir / "metadata.json").read_text())
    assert meta["features"] == FEATURE_NAMES
    m = meta["metrics"]
    assert m["xgboost"]["roc_auc"] > m["rules"]["roc_auc"]
    assert m["xgboost"]["pr_auc"] > m["rules"]["pr_auc"]
    op = m["xgboost"]["operating_point"]
    assert 0 < op["fpr"] < 0.03  # threshold chosen on validation transfers to test
    ref = json.loads((model_dir / "reference.json").read_text())
    assert "__score__" in ref and set(FEATURE_NAMES) <= set(ref)
    bank = pd.read_parquet(model_dir / "case_bank.parquet")
    assert bank.is_fraud.any() and (~bank.is_fraud).any()


def test_registry_promote_and_history(tmp_path: Path, model_dir: Path):
    reg = ModelRegistry(str(tmp_path / "reg"))
    assert reg.champion() is None
    reg.publish(model_dir, "v1")
    reg.promote("v1", "first")
    reg.publish(model_dir, "v2")
    doc = reg.promote("v2", "better", {"recall": 0.9})
    assert reg.champion()["version"] == "v2" and doc["previous"] == "v1"
    assert [h["version"] for h in reg.champion()["history"]] == ["v1", "v2"]
    assert reg.list_versions() == ["v1", "v2"]
    assert reg.version_marker() is not None
    fetched = reg.fetch("v2", tmp_path / "cache")
    assert (fetched / "model.json").exists()


def test_challenger_decision_rules():
    champ = {"recall_at_1.000%_fpr": 0.80, "pr_auc": 0.70}
    assert decide(None, champ, 0.01)[0]
    assert not decide(champ, {"recall_at_1.000%_fpr": 0.805, "pr_auc": 0.71}, 0.01)[0]
    assert not decide(champ, {"recall_at_1.000%_fpr": 0.85, "pr_auc": 0.60}, 0.01)[0]
    ok, why = decide(champ, {"recall_at_1.000%_fpr": 0.85, "pr_auc": 0.72}, 0.01)
    assert ok and "0.800 -> 0.850" in why
