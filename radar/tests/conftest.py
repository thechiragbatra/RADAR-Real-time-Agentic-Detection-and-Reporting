"""Shared fixtures: a small simulated world and a model trained on it (built once per session)."""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import pytest

os.environ.setdefault("AWS_DEFAULT_REGION", "ap-south-1")
os.environ.setdefault("AWS_ACCESS_KEY_ID", "testing")
os.environ.setdefault("AWS_SECRET_ACCESS_KEY", "testing")

from radar.model.registry import ModelRegistry  # noqa: E402
from radar.model.train import train_and_package  # noqa: E402
from radar.simulator.generate import run as simulate  # noqa: E402


@pytest.fixture(scope="session")
def data_dir(tmp_path_factory) -> Path:
    d = tmp_path_factory.mktemp("data")
    simulate(d, n_users=700, n_merchants=250, days=40, seed=7, fraud_user_fraction=0.08)
    return d


@pytest.fixture(scope="session")
def transactions(data_dir: Path) -> pd.DataFrame:
    return pd.read_parquet(data_dir / "transactions.parquet")


@pytest.fixture(scope="session")
def models_dir(tmp_path_factory, data_dir: Path) -> Path:
    m = tmp_path_factory.mktemp("models")
    vdir, meta = train_and_package(
        data_dir, m, burn_in_days=7, val_days=7, test_days=7, version="vtest"
    )
    reg = ModelRegistry(str(m))
    reg.publish(vdir, "vtest")
    reg.promote("vtest", "test fixture")
    return m


@pytest.fixture(scope="session")
def model_dir(models_dir: Path) -> Path:
    return models_dir / "vtest"
