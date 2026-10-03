"""Generate a labelled transaction dataset.

    radar-simulate --users 15000 --days 90 --out data/

Writes ``transactions.parquet`` (time-ordered, labelled), ``users.parquet``,
``merchants.parquet`` and ``meta.json``.

Legitimate behaviour includes the things that make real fraud detection hard: travel,
phone upgrades (new device + electronics purchase), sale-day shopping sprees, monthly
bill-pay bursts, night owls and family members sharing a phone. Without these, any
model looks perfect and the evaluation is meaningless.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from radar.schemas import Channel, FraudType
from radar.simulator.fraud_patterns import PATTERNS, make_txn
from radar.simulator.world import (
    INDIAN_CITIES,
    IST_OFFSET,
    User,
    World,
    build_world,
    draw_amount,
    sample_time_in_day,
)


def _poisson(rng: random.Random, lam: float) -> int:
    if lam <= 0:
        return 0
    if lam > 30:  # normal approximation, never hit with default rates
        return max(0, int(rng.gauss(lam, math.sqrt(lam)) + 0.5))
    limit, k, p = math.exp(-lam), 0, 1.0
    while True:
        p *= rng.random()
        if p <= limit:
            return k
        k += 1


def _pick_device(
    rng: random.Random, user: User, day: int, upgrade_day: int | None, new_device: str
) -> str:
    if upgrade_day is not None and day > upgrade_day and rng.random() < 0.85:
        return new_device
    r = rng.random()
    if r < 0.78 or len(user.devices) == 1:
        return user.devices[0]
    if r < 0.95 or len(user.devices) == 2:
        return user.devices[1]
    return user.devices[2]  # the shared family phone


def _legit_user_txns(world: World, user: User) -> list[dict]:
    rng = world.rng
    txns: list[dict] = []
    home_pool = world.merchants_by_city[user.home_city]
    online = world.online_merchants

    # ---- one-off life events (the hard negatives) ---------------------------
    trip: tuple[int, int, str] | None = None
    if world.days >= 14 and rng.random() < 0.22:
        dest = rng.choice([c for c in INDIAN_CITIES if c[0] != user.home_city])
        d0 = rng.randint(5, world.days - 6)
        trip = (d0, d0 + rng.randint(2, 6), dest[0])
        travel_merchants = [m for m in online if world.merchants[m].category == "travel"]
        if travel_merchants:
            book_day = max(0, d0 - rng.randint(1, 12))
            ts = sample_time_in_day(rng, world.start_ts + book_day * 86400, user.hour_weights)
            txns.append(
                make_txn(
                    world,
                    user,
                    ts,
                    rng.choice(travel_merchants),
                    user.devices[0],
                    Channel.CARD,
                    draw_amount(rng, "travel", user.spend_level),
                    context="trip_booking",
                )
            )
    upgrade_day: int | None = (
        rng.randint(3, world.days - 3) if (world.days >= 7 and rng.random() < 0.14) else None
    )
    new_device = f"{user.devices[0]}n"
    big_purchase_day: int | None = rng.randint(0, world.days - 1) if rng.random() < 0.3 else None
    spree_day: int | None = rng.randint(0, world.days - 1) if rng.random() < 0.35 else None
    billpay_days = {d for d in range(1, world.days, 30) if rng.random() < 0.6}

    for day in range(world.days):
        day_start = world.start_ts + day * 86400
        n = _poisson(rng, user.daily_rate)
        on_trip = trip is not None and trip[0] <= day <= trip[1]
        for _ in range(n):
            ts = sample_time_in_day(rng, day_start, user.hour_weights)
            if on_trip and trip is not None:
                dest_pool = world.merchants_by_city[trip[2]]
                mid = (
                    rng.choice(dest_pool)
                    if (rng.random() < 0.7 and dest_pool)
                    else rng.choice(online)
                )
            elif rng.random() < 0.8:
                mid = rng.choice(user.favourite_merchants)
            else:
                mid = rng.choice(home_pool if (rng.random() < 0.5 and home_pool) else online)
            cat = world.merchants[mid].category
            device = _pick_device(rng, user, day, upgrade_day, new_device)
            channel = rng.choices(list(Channel), weights=user.channel_weights)[0]
            amount = draw_amount(rng, cat, user.spend_level)
            loc = None
            if on_trip and trip is not None and world.merchants[mid].is_online:
                dest = next(c for c in INDIAN_CITIES if c[0] == trip[2])
                loc = (dest[0], dest[1], dest[2])
            txns.append(
                make_txn(
                    world,
                    user,
                    ts,
                    mid,
                    device,
                    channel,
                    amount,
                    city=loc[0] if loc else None,
                    lat=loc[1] if loc else None,
                    lon=loc[2] if loc else None,
                    context="trip" if on_trip else None,
                )
            )

        # phone upgrade: buy a phone on the old device, then the new device shows up
        if upgrade_day == day:
            pool = [m for m in home_pool if world.merchants[m].category == "electronics"] or [
                m for m in online if world.merchants[m].category == "electronics"
            ]
            if pool:
                ts = day_start + rng.uniform(11, 20) * 3600 - IST_OFFSET
                txns.append(
                    make_txn(
                        world,
                        user,
                        ts,
                        rng.choice(pool),
                        user.devices[0],
                        Channel.CARD,
                        draw_amount(
                            rng, "electronics", user.spend_level, multiplier=rng.uniform(1.0, 1.8)
                        ),
                        context="phone_upgrade",
                    )
                )
                # first few purchases on the new phone within the next hour: apps, recharge, food
                t = ts + rng.uniform(600, 3600)
                for _ in range(rng.randint(1, 3)):
                    mid = rng.choice(
                        [
                            m
                            for m in online
                            if world.merchants[m].category
                            in ("mobile_recharge", "entertainment", "food_delivery")
                        ]
                        or online
                    )
                    txns.append(
                        make_txn(
                            world,
                            user,
                            t,
                            mid,
                            new_device,
                            Channel.UPI,
                            draw_amount(rng, world.merchants[mid].category, user.spend_level),
                            context="phone_upgrade",
                        )
                    )
                    t += rng.uniform(120, 900)

        if big_purchase_day == day:
            cat = rng.choice(["electronics", "jewellery", "electronics", "apparel", "travel"])
            pool = [m for m in home_pool if world.merchants[m].category == cat] or [
                m for m in online if world.merchants[m].category == cat
            ]
            if pool:
                ts = day_start + rng.uniform(11, 20) * 3600 - IST_OFFSET
                txns.append(
                    make_txn(
                        world,
                        user,
                        ts,
                        rng.choice(pool),
                        _pick_device(rng, user, day, upgrade_day, new_device),
                        rng.choice([Channel.CARD, Channel.NETBANKING]),
                        draw_amount(rng, cat, user.spend_level, multiplier=rng.uniform(1.0, 2.5)),
                        context="big_purchase",
                    )
                )

        # sale-day spree: several purchases in an hour or two, mostly online
        if spree_day == day:
            t = day_start + rng.uniform(10, 22) * 3600 - IST_OFFSET
            for _ in range(rng.randint(4, 8)):
                mid = rng.choice(
                    [
                        m
                        for m in online
                        if world.merchants[m].category
                        in ("ecommerce", "apparel", "electronics", "entertainment")
                    ]
                    or online
                )
                cat = world.merchants[mid].category
                txns.append(
                    make_txn(
                        world,
                        user,
                        t,
                        mid,
                        _pick_device(rng, user, day, upgrade_day, new_device),
                        rng.choices(list(Channel), weights=user.channel_weights)[0],
                        draw_amount(rng, cat, user.spend_level, multiplier=rng.uniform(0.8, 2.2)),
                        context="sale_spree",
                    )
                )
                t += rng.uniform(180, 1500)

        # monthly bill-pay: utilities + recharges back to back
        if day in billpay_days:
            t = day_start + rng.uniform(8, 22) * 3600 - IST_OFFSET
            for _ in range(rng.randint(2, 4)):
                mid = rng.choice(
                    [
                        m
                        for m in online
                        if world.merchants[m].category in ("utilities", "mobile_recharge")
                    ]
                    or online
                )
                txns.append(
                    make_txn(
                        world,
                        user,
                        t,
                        mid,
                        _pick_device(rng, user, day, upgrade_day, new_device),
                        Channel.UPI if rng.random() < 0.6 else Channel.NETBANKING,
                        draw_amount(rng, world.merchants[mid].category, user.spend_level),
                        context="bill_pay",
                    )
                )
                t += rng.uniform(60, 600)
    return txns


def generate(world: World, fraud_user_fraction: float) -> pd.DataFrame:
    rng = world.rng
    all_txns: list[dict] = []
    per_user_legit: dict[str, list[dict]] = {}
    for user in world.users:
        legit = _legit_user_txns(world, user)
        per_user_legit[user.user_id] = legit
        all_txns.extend(legit)

    # victims are drawn in proportion to activity: an account that transacts more is
    # exposed more (more merchants holding its card, more sessions to hijack). Sampling
    # uniformly instead would make "low activity" a leak the model happily exploits.
    n_fraud_users = int(len(world.users) * fraud_user_fraction)
    victims: list[User] = []
    seen: set[str] = set()
    rates = [u.daily_rate for u in world.users]
    while len(victims) < n_fraud_users:
        u = rng.choices(world.users, weights=rates)[0]
        if u.user_id not in seen:
            seen.add(u.user_id)
            victims.append(u)
    names = list(PATTERNS)
    weights = [PATTERNS[n][1] for n in names]
    for user in victims:
        ftype = rng.choices(names, weights=weights)[0]
        fn = PATTERNS[ftype][0]
        if ftype == FraudType.GEO_JUMP:
            legit = [t for t in per_user_legit[user.user_id] if t["city"] == user.home_city]
            if not legit:
                continue
            anchor = rng.choice(legit)
            episode = fn(world, user, anchor["ts"])
        else:
            t0 = world.start_ts + rng.uniform(1, world.days - 1) * 86400
            episode = fn(world, user, t0)
        all_txns.extend(t for t in episode if world.start_ts <= t["ts"] < world.end_ts)

    df = pd.DataFrame(all_txns)
    df = df.sort_values("ts", kind="stable").reset_index(drop=True)
    df.insert(0, "txn_id", [f"T{i:09d}" for i in range(len(df))])
    df["ts"] = df["ts"].round(3)
    return df


def world_tables(world: World) -> tuple[pd.DataFrame, pd.DataFrame]:
    users = pd.DataFrame(
        {
            "user_id": u.user_id,
            "home_city": u.home_city,
            "home_lat": u.home_lat,
            "home_lon": u.home_lon,
            "account_created_ts": u.account_created_ts,
            "spend_level": u.spend_level,
        }
        for u in world.users
    )
    merchants = pd.DataFrame(
        {
            "merchant_id": m.merchant_id,
            "category": m.category,
            "city": m.city,
            "country": m.country,
            "is_online": m.is_online,
            "created_ts": m.created_ts,
            "category_risk": m.category_risk,
        }
        for m in world.merchants.values()
    )
    return users, merchants


def run(
    out: Path,
    n_users: int = 15000,
    n_merchants: int = 1500,
    days: int = 90,
    seed: int = 42,
    fraud_user_fraction: float = 0.05,
    start: str = "2026-06-01",
) -> dict:
    t0 = time.time()
    start_ts = datetime.fromisoformat(start).replace(tzinfo=UTC).timestamp()
    world = build_world(n_users, n_merchants, days, start_ts, seed)
    df = generate(world, fraud_user_fraction)
    users, merchants = world_tables(world)
    out.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out / "transactions.parquet", index=False)
    users.to_parquet(out / "users.parquet", index=False)
    merchants.to_parquet(out / "merchants.parquet", index=False)
    meta = {
        "seed": seed,
        "users": n_users,
        "merchants": len(merchants),
        "days": days,
        "start_ts": start_ts,
        "end_ts": world.end_ts,
        "transactions": int(len(df)),
        "fraud_transactions": int(df["is_fraud"].sum()),
        "fraud_rate": float(df["is_fraud"].mean()),
        "fraud_by_type": {
            k: int(v) for k, v in df[df.is_fraud]["fraud_type"].value_counts().items()
        },
        "generation_seconds": round(time.time() - t0, 1),
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    return meta


def main() -> None:
    p = argparse.ArgumentParser(description="Generate synthetic labelled transactions")
    p.add_argument("--users", type=int, default=15000)
    p.add_argument("--merchants", type=int, default=1500)
    p.add_argument("--days", type=int, default=90)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--fraud-user-fraction", type=float, default=0.05)
    p.add_argument("--start", default="2026-06-01")
    p.add_argument("--out", type=Path, default=Path("data"))
    a = p.parse_args()
    meta = run(a.out, a.users, a.merchants, a.days, a.seed, a.fraud_user_fraction, a.start)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
