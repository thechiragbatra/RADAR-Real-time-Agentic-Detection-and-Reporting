import math

import boto3
import numpy as np
import pandas as pd
import pytest
from moto import mock_aws

from radar.features import (
    FEATURE_NAMES,
    DeviceState,
    DynamoDBFeatureStore,
    InMemoryFeatureStore,
    LocalReferenceData,
    UserState,
    build_training_frame,
    compute_features,
    featurize_and_update,
)
from radar.schemas import Channel, MerchantProfile, Transaction, UserProfile

T0 = 1_780_272_000.0  # 2026-06-01 00:00 UTC


def txn(
    i: int,
    ts: float,
    amount: float,
    device="D1",
    merchant="M1",
    city="Delhi",
    lat=28.6,
    lon=77.2,
    country="IN",
    online=False,
):
    return Transaction(
        txn_id=f"T{i}",
        ts=ts,
        user_id="U1",
        merchant_id=merchant,
        merchant_category="grocery",
        device_id=device,
        channel=Channel.UPI,
        amount=amount,
        city=city,
        lat=lat,
        lon=lon,
        country=country,
        is_online=online,
    )


PROFILE = UserProfile(
    user_id="U1",
    home_city="Delhi",
    home_lat=28.6,
    home_lon=77.2,
    account_created_ts=T0 - 400 * 86400,
)
MERCHANT = MerchantProfile(
    merchant_id="M1",
    category="grocery",
    city="Delhi",
    is_online=False,
    created_ts=T0 - 900 * 86400,
    category_risk=0.02,
)


def test_cold_start_defaults():
    f = compute_features(txn(0, T0, 500), UserState("U1"), DeviceState("D1"), PROFILE, MERCHANT)
    assert set(f) == set(FEATURE_NAMES)
    assert f["is_new_device"] == 1 and f["is_new_merchant"] == 1 and f["txn_count_total"] == 0
    assert f["amount_z"] == 0 and f["seconds_since_last"] == 30 * 86400
    assert f["account_age_days"] == pytest.approx(400)
    assert f["is_home_city"] == 1 and f["is_international"] == 0


def test_welford_matches_numpy_and_zscore():
    s = UserState("U1")
    amounts = [120, 340, 90, 800, 410, 220, 150]
    for i, a in enumerate(amounts):
        s.update(txn(i, T0 + i * 3600, a))
    assert s.amount_mean == pytest.approx(np.mean(amounts))
    assert s.amount_std == pytest.approx(np.std(amounts, ddof=1))
    f = compute_features(txn(99, T0 + 8 * 3600, 5000), s, DeviceState("D1"), PROFILE, MERCHANT)
    assert f["amount_z"] == pytest.approx((5000 - np.mean(amounts)) / np.std(amounts, ddof=1))
    assert f["is_new_device"] == 0 and f["is_new_merchant"] == 0
    assert f["device_user_age_days"] == pytest.approx(8 / 24)


def test_velocity_windows_and_small_txn_count():
    s = UserState("U1")
    base = T0 + 12 * 3600
    for i, (dt, a) in enumerate([(-7200, 500), (-3000, 20), (-1800, 30), (-600, 40), (-100, 2000)]):
        s.update(txn(i, base + dt, a))
    f = compute_features(txn(9, base, 100), s, DeviceState("D1"), PROFILE, MERCHANT)
    assert f["txn_count_1h"] == 4 and f["txn_count_24h"] == 5
    assert f["small_txn_count_1h"] == 3
    assert f["amount_sum_24h"] == pytest.approx(2590)
    assert f["seconds_since_last"] == 100


def test_impossible_travel_speed():
    s = UserState("U1")
    s.update(txn(0, T0, 300, lat=28.6, lon=77.2))  # Delhi
    f = compute_features(
        txn(1, T0 + 1800, 300, city="Mumbai", lat=19.07, lon=72.87),
        s,
        DeviceState("D1"),
        PROFILE,
        MERCHANT,
    )
    assert 1100 < f["distance_from_last_km"] < 1200
    assert f["speed_kmh"] > 2000
    assert f["is_home_city"] == 0


def test_device_sharing_signal():
    d = DeviceState("DX")
    d.update(Transaction(**{**txn(0, T0, 10).model_dump(), "user_id": "U7"}))
    d.update(Transaction(**{**txn(1, T0 + 10, 10).model_dump(), "user_id": "U8"}))
    f = compute_features(txn(2, T0 + 20, 10, device="DX"), UserState("U1"), d, PROFILE, MERCHANT)
    assert f["device_user_count"] == 3  # two others + this user


def test_night_and_weekend_use_ist():
    # 2026-06-06 is a Saturday; 22:00 UTC = 03:30 IST Sunday
    sat_2200_utc = T0 + 5 * 86400 + 22 * 3600
    f = compute_features(
        txn(0, sat_2200_utc, 10), UserState("U1"), DeviceState("D1"), PROFILE, MERCHANT
    )
    assert (
        f["is_night"] == 1
        and f["is_weekend"] == 1
        and math.isclose(f["hour_local"], 3.5, abs_tol=0.01)
    )


def test_state_is_bounded_and_serialisable():
    s = UserState("U1")
    for i in range(700):
        s.update(txn(i, T0 + i * 60, 10, device=f"D{i}", merchant=f"M{i}"))
    assert len(s.recent) <= 500 and len(s.known_devices) <= 50 and len(s.known_merchants) <= 300
    rt = UserState.from_dict(s.to_dict())
    assert rt == s


def test_replay_has_no_leakage_and_matches_online_path(data_dir, transactions):
    ref = LocalReferenceData.from_dir(data_dir)
    sub = transactions.head(3000)
    offline = build_training_frame(sub, ref)
    store = InMemoryFeatureStore()
    rows = []
    for row in sub.itertuples(index=False):
        d = {k: getattr(row, k) for k in Transaction.model_fields}
        rows.append(featurize_and_update(Transaction(**d), store, ref))
    online = pd.DataFrame(rows)[FEATURE_NAMES]
    pd.testing.assert_frame_equal(
        offline[FEATURE_NAMES].reset_index(drop=True), online, check_dtype=False
    )
    assert offline.isna().sum().sum() == 0
    assert list(offline.columns[:3]) == ["txn_id", "ts", "user_id"]


@mock_aws
def test_dynamodb_store_roundtrip():
    ddb = boto3.resource("dynamodb", region_name="ap-south-1")
    ddb.create_table(
        TableName="state",
        KeySchema=[{"AttributeName": "pk", "KeyType": "HASH"}],
        AttributeDefinitions=[{"AttributeName": "pk", "AttributeType": "S"}],
        BillingMode="PAY_PER_REQUEST",
    )
    store = DynamoDBFeatureStore("state", resource=ddb)
    assert store.get_user_state("U1").txn_count == 0
    s = UserState("U1")
    s.update(txn(0, T0, 123.45, device="D9"))
    store.put_user_state(s)
    d = DeviceState("D9")
    d.update(txn(0, T0, 123.45, device="D9"))
    store.put_device_state(d)
    assert store.get_user_state("U1") == s
    assert store.get_device_state("D9") == d
