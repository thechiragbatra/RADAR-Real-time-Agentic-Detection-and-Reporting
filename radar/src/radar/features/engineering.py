"""Feature definitions.

Every feature is computed from (a) the transaction itself, (b) the user's and device's
rolling state *before* this transaction, and (c) static reference data. Nothing here can
see the future or the label, and the function is identical in training and serving.
"""

from __future__ import annotations

import math
from collections.abc import Iterable

import pandas as pd

from radar.features.reference import LocalReferenceData, ReferenceData
from radar.features.state import DeviceState, UserState
from radar.features.store import FeatureStore, InMemoryFeatureStore
from radar.schemas import Channel, MerchantProfile, Transaction, UserProfile

IST_OFFSET = 5.5 * 3600
DAY = 86400.0
MAX_GAP = 30 * DAY

CATEGORY_LIST = [
    "grocery", "food_delivery", "fuel", "utilities", "mobile_recharge", "pharmacy",
    "apparel", "ecommerce", "entertainment", "electronics", "travel", "gaming",
    "gift_cards", "jewellery", "forex", "crypto",
]  # fmt: skip

FEATURE_NAMES: list[str] = [
    "amount",
    "amount_log",
    "amount_z",
    "amount_ratio_mean",
    "txn_count_1h",
    "txn_count_24h",
    "amount_sum_24h",
    "small_txn_count_1h",
    "seconds_since_last",
    "is_new_device",
    "device_user_age_days",
    "is_new_merchant",
    "user_device_count",
    "device_user_count",
    "device_age_days",
    "distance_from_last_km",
    "speed_kmh",
    "hour_local",
    "is_night",
    "is_weekend",
    "is_online",
    "is_international",
    "is_home_city",
    "account_age_days",
    "txn_count_total",
    "category_risk",
    "merchant_age_days",
    "channel_card",
    "channel_netbanking",
    *[f"cat_{c}" for c in CATEGORY_LIST],
]


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(min(1.0, a)))


def compute_features(
    txn: Transaction,
    user: UserState,
    device: DeviceState,
    profile: UserProfile | None,
    merchant: MerchantProfile | None,
) -> dict[str, float]:
    amt = txn.amount
    last_1h = user.recent_within(txn.ts, 3600)
    last_24h = user.recent_within(txn.ts, DAY)

    if user.txn_count >= 3 and user.amount_std > 0:
        z = (amt - user.amount_mean) / user.amount_std
    else:
        z = 0.0
    ratio = amt / user.amount_mean if user.amount_mean > 0 else 1.0

    if user.last_ts is None:
        gap = MAX_GAP
        dist = 0.0
    else:
        gap = min(MAX_GAP, max(0.0, txn.ts - user.last_ts))
        dist = (
            _haversine_km(user.last_lat, user.last_lon, txn.lat, txn.lon)
            if user.last_lat is not None and user.last_lon is not None
            else 0.0
        )
    speed = min(5000.0, dist / max(gap / 3600, 1 / 60)) if dist > 0 else 0.0

    local_seconds = (txn.ts + IST_OFFSET) % DAY
    hour = local_seconds / 3600
    weekday = int(((txn.ts + IST_OFFSET) // DAY + 3) % 7)  # 1970-01-01 was a Thursday (3)

    first_with_user = user.known_devices.get(txn.device_id)
    is_new_device = float(first_with_user is None)
    device_user_age = (txn.ts - first_with_user) / DAY if first_with_user is not None else 0.0
    device_age = (txn.ts - device.first_seen_ts) / DAY if device.first_seen_ts is not None else 0.0
    device_user_count = float(len(device.users) + (txn.user_id not in device.users))

    account_age = (
        (txn.ts - profile.account_created_ts) / DAY
        if profile is not None
        else (txn.ts - user.first_ts) / DAY
        if user.first_ts
        else 0.0
    )
    home = profile.home_city if profile is not None else (user.last_city or txn.city)

    f: dict[str, float] = {
        "amount": amt,
        "amount_log": math.log1p(amt),
        "amount_z": max(-10.0, min(50.0, z)),
        "amount_ratio_mean": min(100.0, ratio),
        "txn_count_1h": float(len(last_1h)),
        "txn_count_24h": float(len(last_24h)),
        "amount_sum_24h": float(sum(a for _, a in last_24h)),
        "small_txn_count_1h": float(sum(1 for _, a in last_1h if a < 100)),
        "seconds_since_last": gap,
        "is_new_device": is_new_device,
        "device_user_age_days": max(0.0, device_user_age),
        "is_new_merchant": float(txn.merchant_id not in user.known_merchants),
        "user_device_count": float(len(user.known_devices) + is_new_device),
        "device_user_count": device_user_count,
        "device_age_days": max(0.0, device_age),
        "distance_from_last_km": dist,
        "speed_kmh": speed,
        "hour_local": hour,
        "is_night": float(hour < 5.5),
        "is_weekend": float(weekday >= 5),
        "is_online": float(txn.is_online),
        "is_international": float(txn.country != "IN"),
        "is_home_city": float(txn.city == home),
        "account_age_days": max(0.0, account_age),
        "txn_count_total": float(user.txn_count),
        "category_risk": merchant.category_risk if merchant is not None else 0.1,
        "merchant_age_days": max(0.0, (txn.ts - merchant.created_ts) / DAY)
        if merchant is not None
        else 0.0,
        "channel_card": float(txn.channel == Channel.CARD),
        "channel_netbanking": float(txn.channel == Channel.NETBANKING),
    }
    for c in CATEGORY_LIST:
        f[f"cat_{c}"] = float(txn.merchant_category == c)
    return f


def featurize_and_update(
    txn: Transaction, store: FeatureStore, ref: ReferenceData
) -> dict[str, float]:
    """Compute features for ``txn`` then fold it into the user's and device's state."""
    ustate = store.get_user_state(txn.user_id)
    dstate = store.get_device_state(txn.device_id)
    feats = compute_features(
        txn, ustate, dstate, ref.get_user_profile(txn.user_id), ref.get_merchant(txn.merchant_id)
    )
    ustate.update(txn)
    dstate.update(txn)
    store.put_user_state(ustate)
    store.put_device_state(dstate)
    return feats


def _iter_transactions(df: pd.DataFrame) -> Iterable[Transaction]:
    cols = list(Transaction.model_fields)
    for row in df[cols].itertuples(index=False):
        d = dict(zip(cols, row, strict=True))
        d["channel"] = Channel(d["channel"])
        d["is_online"] = bool(d["is_online"])
        yield Transaction.model_construct(**d)


def build_training_frame(
    transactions: pd.DataFrame,
    ref: LocalReferenceData,
    store: FeatureStore | None = None,
) -> pd.DataFrame:
    """Replay a time-ordered transaction file through the feature pipeline.

    Returns one row per transaction with ``FEATURE_NAMES`` plus ``txn_id``, ``ts``,
    ``user_id`` and, if present, ``is_fraud`` / ``fraud_type``.
    """
    if not transactions["ts"].is_monotonic_increasing:
        transactions = transactions.sort_values("ts", kind="stable")
    store = store or InMemoryFeatureStore()
    rows = [featurize_and_update(t, store, ref) for t in _iter_transactions(transactions)]
    feats = pd.DataFrame(rows, columns=FEATURE_NAMES)
    out = transactions[["txn_id", "ts", "user_id"]].reset_index(drop=True).join(feats)
    for col in ("is_fraud", "fraud_type"):
        if col in transactions.columns:
            out[col] = transactions[col].to_numpy()
    return out
