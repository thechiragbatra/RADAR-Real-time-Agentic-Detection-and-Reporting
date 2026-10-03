import json
from pathlib import Path

import pandas as pd

from radar.schemas import FraudType, Transaction
from radar.simulator.generate import run as simulate
from radar.simulator.world import haversine_km


def test_dataset_shape_and_labels(data_dir: Path, transactions: pd.DataFrame):
    meta = json.loads((data_dir / "meta.json").read_text())
    assert meta["transactions"] == len(transactions) > 10_000
    assert 0.002 < transactions.is_fraud.mean() < 0.05
    assert transactions.ts.is_monotonic_increasing
    assert transactions.txn_id.is_unique
    present = set(transactions[transactions.is_fraud].fraud_type.unique())
    assert {t.value for t in FraudType if t != FraudType.NONE} <= present


def test_wire_format_has_no_labels(transactions: pd.DataFrame):
    row = transactions.iloc[0].to_dict()
    txn = Transaction.model_validate({k: row[k] for k in Transaction.model_fields})
    assert not hasattr(txn, "is_fraud")
    assert txn.amount > 0 and txn.channel.value in {"UPI", "CARD", "NETBANKING"}


def test_generation_is_deterministic(tmp_path: Path):
    a = simulate(tmp_path / "a", n_users=120, n_merchants=60, days=10, seed=3)
    b = simulate(tmp_path / "b", n_users=120, n_merchants=60, days=10, seed=3)
    assert (
        a["transactions"] == b["transactions"]
        and a["fraud_transactions"] == b["fraud_transactions"]
    )
    da = pd.read_parquet(tmp_path / "a/transactions.parquet")
    db = pd.read_parquet(tmp_path / "b/transactions.parquet")
    pd.testing.assert_frame_equal(da, db)


def test_hard_negatives_exist(transactions: pd.DataFrame):
    legit = transactions[~transactions.is_fraud]
    for ctx in ("trip", "phone_upgrade", "sale_spree", "bill_pay"):
        assert (legit.context == ctx).sum() > 0, ctx


def test_geo_jump_is_far_from_home(transactions: pd.DataFrame, data_dir: Path):
    users = pd.read_parquet(data_dir / "users.parquet").set_index("user_id")
    gj = transactions[transactions.fraud_type == "geo_jump"].head(20)
    for r in gj.itertuples():
        u = users.loc[r.user_id]
        assert haversine_km(u.home_lat, u.home_lon, r.lat, r.lon) > 250


def test_card_testing_is_small_and_fast(transactions: pd.DataFrame):
    ct = transactions[transactions.fraud_type == "card_testing"]
    small = ct[ct.amount < 100]
    assert len(small) / len(ct) > 0.7
    # episodes are concentrated in time: median gap between consecutive card-testing txns of a user is minutes
    gaps = ct.sort_values("ts").groupby("user_id").ts.diff().dropna()
    assert gaps.median() < 600
