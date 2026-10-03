"""Fraud episode generators.

Each function receives the world, a victim and a start time and returns a list of raw
transaction dicts. They deliberately differ in *which* signal gives them away:

    account_takeover    unfamiliar device, often at night, higher-risk online merchants
    card_testing        burst of tiny amounts in minutes, sometimes followed by a large one
    velocity_burst      the user's own device - only velocity and amount deviate
    geo_jump            implausible travel relative to the previous legit transaction
    merchant_collusion  a few larger purchases at a high-risk merchant, days apart
    low_and_slow        one modest online purchase a day from a cloned device for a week

Multipliers are kept modest on purpose. Fraud that is 10x a user's normal spend is caught
by a single rule; the interesting cases sit inside the normal range and are only visible
through combinations of weak signals. A detector that only learns one pattern will look
great on average and miss the rest - which is why evaluation reports recall per type.
"""

from __future__ import annotations

import random

from radar.schemas import Channel, FraudType
from radar.simulator.world import (
    CARD_TEST_TARGETS,
    COLLUSION_TARGETS,
    HIGH_RISK_ONLINE,
    INDIAN_CITIES,
    IST_OFFSET,
    User,
    World,
    draw_amount,
    haversine_km,
)


def make_txn(
    world: World,
    user: User,
    ts: float,
    merchant_id: str,
    device_id: str,
    channel: Channel,
    amount: float,
    *,
    city: str | None = None,
    lat: float | None = None,
    lon: float | None = None,
    country: str = "IN",
    fraud_type: FraudType = FraudType.NONE,
    context: str | None = None,
) -> dict:
    m = world.merchants[merchant_id]
    if m.is_online:
        # an online purchase is located wherever the payer is
        loc_city = city or user.home_city
        loc_lat = lat if lat is not None else user.home_lat
        loc_lon = lon if lon is not None else user.home_lon
    else:
        loc_city, loc_lat, loc_lon = m.city, m.lat, m.lon
        country = m.country
    return {
        "ts": ts,
        "user_id": user.user_id,
        "merchant_id": merchant_id,
        "merchant_category": m.category,
        "device_id": device_id,
        "channel": channel.value,
        "amount": amount,
        "city": loc_city,
        "lat": round(loc_lat, 5),
        "lon": round(loc_lon, 5),
        "country": country,
        "is_online": m.is_online,
        "is_fraud": fraud_type != FraudType.NONE,
        "fraud_type": fraud_type.value,
        # why this transaction exists (simulator bookkeeping, never a feature)
        "context": context or (fraud_type.value if fraud_type != FraudType.NONE else "regular"),
    }


def _online_merchant_in(world: World, rng: random.Random, categories: list[str]) -> str:
    pool = [m for m in world.online_merchants if world.merchants[m].category in categories]
    return rng.choice(pool or world.online_merchants)


def _attacker_device(world: World, rng: random.Random, p_ring: float = 0.5) -> str:
    """Half the attacks reuse a fraud-ring device, half use a fresh one-off fingerprint."""
    if rng.random() < p_ring:
        return rng.choice(world.ring_devices)
    return f"DX{rng.randrange(10**8):08d}"


def _night_time(rng: random.Random, t0: float, p: float = 0.5) -> float:
    """With probability p, shift t0 to between 00:30 and 05:00 local on the same day."""
    if rng.random() >= p:
        return t0
    local_day_start = t0 - ((t0 + IST_OFFSET) % 86400)
    return local_day_start + rng.uniform(0.5, 5.0) * 3600


def account_takeover(world: World, user: User, t0: float) -> list[dict]:
    rng = world.rng
    t = _night_time(rng, t0)
    device = _attacker_device(world, rng)
    other_city = rng.choice([c for c in INDIAN_CITIES if c[0] != user.home_city])
    use_other = rng.random() < 0.5
    txns = []
    for i in range(rng.randint(2, 5)):
        # first purchase often "tests" a familiar merchant with a card on file
        if i == 0 and rng.random() < 0.4:
            mid = rng.choice(
                [m for m in user.favourite_merchants if world.merchants[m].is_online]
                or world.online_merchants
            )
        else:
            mid = _online_merchant_in(world, rng, HIGH_RISK_ONLINE)
        cat = world.merchants[mid].category
        amt = draw_amount(rng, cat, user.spend_level, multiplier=rng.uniform(1.3, 4.0))
        txns.append(
            make_txn(
                world,
                user,
                t,
                mid,
                device,
                rng.choice([Channel.CARD, Channel.NETBANKING, Channel.UPI]),
                amt,
                city=other_city[0] if use_other else None,
                lat=other_city[1] if use_other else None,
                lon=other_city[2] if use_other else None,
                fraud_type=FraudType.ACCOUNT_TAKEOVER,
            )
        )
        t += rng.uniform(120, 2400)
    return txns


def card_testing(world: World, user: User, t0: float) -> list[dict]:
    rng = world.rng
    device = _attacker_device(world, rng, p_ring=0.7)
    t = t0
    txns = []
    n_small = rng.randint(4, 12)
    window = rng.uniform(180, 900)
    for _ in range(n_small):
        mid = _online_merchant_in(world, rng, CARD_TEST_TARGETS)
        txns.append(
            make_txn(
                world,
                user,
                t,
                mid,
                device,
                Channel.CARD,
                round(rng.uniform(1, 99), 2),
                fraud_type=FraudType.CARD_TESTING,
            )
        )
        t += window / n_small * rng.uniform(0.3, 1.7)
    if rng.random() < 0.6:  # cash-out does not always follow immediately
        t += rng.uniform(60, 1800)
        mid = _online_merchant_in(world, rng, HIGH_RISK_ONLINE)
        cat = world.merchants[mid].category
        txns.append(
            make_txn(
                world,
                user,
                t,
                mid,
                device,
                Channel.CARD,
                draw_amount(rng, cat, user.spend_level, multiplier=rng.uniform(1.5, 4.0)),
                fraud_type=FraudType.CARD_TESTING,
            )
        )
    return txns


def velocity_burst(world: World, user: User, t0: float) -> list[dict]:
    rng = world.rng
    device = rng.choice(user.devices)
    t = t0
    txns = []
    n = rng.randint(4, 8)
    window = rng.uniform(30, 90) * 60
    home_pool = world.merchants_by_city[user.home_city]
    for _ in range(n):
        if rng.random() < 0.5:
            mid = rng.choice(user.favourite_merchants)
        else:
            mid = rng.choice(
                world.online_merchants if rng.random() < 0.6 or not home_pool else home_pool
            )
        cat = world.merchants[mid].category
        amt = draw_amount(rng, cat, user.spend_level, multiplier=rng.uniform(1.2, 2.5))
        ch = rng.choices(list(Channel), weights=user.channel_weights)[0]
        txns.append(
            make_txn(world, user, t, mid, device, ch, amt, fraud_type=FraudType.VELOCITY_BURST)
        )
        t += window / n * rng.uniform(0.4, 1.6)
    return txns


def geo_jump(world: World, user: User, anchor_ts: float) -> list[dict]:
    """1-3 card-present purchases far away, too soon after a legitimate home transaction."""
    rng = world.rng
    foreign = rng.random() < 0.25
    if foreign:
        city_name = rng.choice(list(world.foreign_merchants_by_city))
    else:
        far = [
            c
            for c in INDIAN_CITIES
            if haversine_km(user.home_lat, user.home_lon, c[1], c[2]) > 300
            and c[0] != user.home_city
        ]
        city_name = rng.choice(far or [c for c in INDIAN_CITIES if c[0] != user.home_city])[0]
    pool = (
        world.foreign_merchants_by_city[city_name]
        if foreign
        else world.merchants_by_city[city_name]
    )
    if not pool:
        pool = world.online_merchants
    device = rng.choice(user.devices) if rng.random() < 0.6 else _attacker_device(world, rng)
    t = anchor_ts + rng.uniform(20, 240) * 60
    txns = []
    for _ in range(rng.randint(1, 3)):
        mid = rng.choice(pool)
        cat = world.merchants[mid].category
        amt = draw_amount(rng, cat, user.spend_level, multiplier=rng.uniform(1.0, 3.0))
        txns.append(
            make_txn(world, user, t, mid, device, Channel.CARD, amt, fraud_type=FraudType.GEO_JUMP)
        )
        t += rng.uniform(600, 3600)
    return txns


def merchant_collusion(world: World, user: User, t0: float) -> list[dict]:
    rng = world.rng
    pool = [
        m
        for m, mm in world.merchants.items()
        if mm.category in COLLUSION_TARGETS
        and m not in user.favourite_merchants
        and mm.country == "IN"
    ]
    mid = rng.choice(pool)
    cat = world.merchants[mid].category
    device = rng.choice(user.devices)
    t = t0
    txns = []
    for _ in range(rng.randint(2, 4)):
        if rng.random() < 0.5:
            amt = float(rng.choice([4999, 5000, 9999, 10000, 14999, 15000, 19999, 24999]))
        else:
            amt = draw_amount(rng, cat, user.spend_level, multiplier=rng.uniform(0.8, 1.5))
        txns.append(
            make_txn(
                world,
                user,
                t,
                mid,
                device,
                rng.choice([Channel.CARD, Channel.NETBANKING]),
                amt,
                fraud_type=FraudType.MERCHANT_COLLUSION,
            )
        )
        t += rng.uniform(1, 4) * 86400
    return txns


def low_and_slow(world: World, user: User, t0: float) -> list[dict]:
    """Cloned-device fraud that stays inside the user's normal spend. The hardest pattern."""
    rng = world.rng
    device = _attacker_device(world, rng, p_ring=0.3)
    t = t0
    txns = []
    fav_online = [m for m in user.favourite_merchants if world.merchants[m].is_online]
    for _ in range(rng.randint(4, 8)):
        if fav_online and rng.random() < 0.5:  # card on file at a merchant the user already uses
            mid = rng.choice(fav_online)
        else:
            mid = _online_merchant_in(
                world, rng, ["ecommerce", "gaming", "entertainment", "apparel", "mobile_recharge"]
            )
        cat = world.merchants[mid].category
        amt = draw_amount(rng, cat, user.spend_level, multiplier=rng.uniform(0.9, 2.0))
        txns.append(
            make_txn(
                world,
                user,
                t,
                mid,
                device,
                rng.choices(list(Channel), weights=user.channel_weights)[0],
                amt,
                fraud_type=FraudType.LOW_AND_SLOW,
            )
        )
        t += rng.uniform(0.6, 1.4) * 86400
    return txns


PATTERNS = {
    FraudType.ACCOUNT_TAKEOVER: (account_takeover, 0.28),
    FraudType.CARD_TESTING: (card_testing, 0.17),
    FraudType.VELOCITY_BURST: (velocity_burst, 0.18),
    FraudType.GEO_JUMP: (geo_jump, 0.17),
    FraudType.MERCHANT_COLLUSION: (merchant_collusion, 0.10),
    FraudType.LOW_AND_SLOW: (low_and_slow, 0.10),
}
