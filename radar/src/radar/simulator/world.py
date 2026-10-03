"""Users, merchants, devices and the geography they live in."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

IST_OFFSET = 5.5 * 3600  # seconds; behaviour is driven by local (IST) time of day

# (city, lat, lon, population weight)
INDIAN_CITIES: list[tuple[str, float, float, float]] = [
    ("Delhi", 28.6139, 77.2090, 16),
    ("Mumbai", 19.0760, 72.8777, 15),
    ("Bengaluru", 12.9716, 77.5946, 12),
    ("Hyderabad", 17.3850, 78.4867, 9),
    ("Chennai", 13.0827, 80.2707, 8),
    ("Kolkata", 22.5726, 88.3639, 7),
    ("Pune", 18.5204, 73.8567, 6),
    ("Ahmedabad", 23.0225, 72.5714, 5),
    ("Jaipur", 26.9124, 75.7873, 4),
    ("Lucknow", 26.8467, 80.9462, 4),
    ("Dehradun", 30.3165, 78.0322, 2),
    ("Chandigarh", 30.7333, 76.7794, 3),
    ("Kochi", 9.9312, 76.2673, 3),
    ("Indore", 22.7196, 75.8577, 3),
    ("Bhopal", 23.2599, 77.4126, 3),
]

FOREIGN_CITIES: list[tuple[str, float, float, str]] = [
    ("Dubai", 25.2048, 55.2708, "AE"),
    ("Singapore", 1.3521, 103.8198, "SG"),
    ("London", 51.5074, -0.1278, "GB"),
    ("Bangkok", 13.7563, 100.5018, "TH"),
    ("Lagos", 6.5244, 3.3792, "NG"),
]

# category -> (typical amount INR, static risk prior, probability merchant is online, popularity weight)
CATEGORIES: dict[str, tuple[float, float, float, float]] = {
    "grocery": (450, 0.02, 0.2, 18),
    "food_delivery": (320, 0.03, 1.0, 16),
    "fuel": (1800, 0.02, 0.0, 8),
    "utilities": (1200, 0.02, 1.0, 6),
    "mobile_recharge": (300, 0.03, 1.0, 6),
    "pharmacy": (350, 0.02, 0.3, 5),
    "apparel": (1500, 0.05, 0.6, 7),
    "ecommerce": (1100, 0.08, 1.0, 14),
    "entertainment": (400, 0.04, 0.9, 6),
    "electronics": (12000, 0.18, 0.7, 4),
    "travel": (6000, 0.12, 0.9, 4),
    "gaming": (500, 0.22, 1.0, 3),
    "gift_cards": (2000, 0.35, 1.0, 2),
    "jewellery": (25000, 0.25, 0.2, 1.5),
    "forex": (15000, 0.30, 0.5, 1),
    "crypto": (8000, 0.40, 1.0, 1),
}

HIGH_RISK_ONLINE = ["electronics", "gift_cards", "crypto", "forex", "gaming", "ecommerce"]
CARD_TEST_TARGETS = ["ecommerce", "gaming", "mobile_recharge", "utilities", "entertainment"]
COLLUSION_TARGETS = ["jewellery", "forex", "crypto", "electronics"]


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = p2 - p1
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


@dataclass
class Merchant:
    merchant_id: str
    category: str
    city: str  # "ONLINE" for pure-online merchants
    lat: float
    lon: float
    is_online: bool
    created_ts: float
    category_risk: float
    country: str = "IN"


@dataclass
class User:
    user_id: str
    home_city: str
    home_lat: float
    home_lon: float
    account_created_ts: float
    spend_level: float
    daily_rate: float
    hour_weights: list[float]  # 24 entries, local time
    devices: list[str]
    favourite_merchants: list[str]
    channel_weights: tuple[float, float, float]  # UPI, CARD, NETBANKING
    extra_devices: list[str] = field(default_factory=list)


@dataclass
class World:
    start_ts: float
    days: int
    users: list[User]
    merchants: dict[str, Merchant]
    merchants_by_city: dict[str, list[str]]
    foreign_merchants_by_city: dict[str, list[str]]
    online_merchants: list[str]
    ring_devices: list[str]
    rng: random.Random

    @property
    def end_ts(self) -> float:
        return self.start_ts + self.days * 86400


def _hour_profile(rng: random.Random) -> list[float]:
    """Mixture of morning / lunch / evening peaks with user-specific weights."""
    peaks = [(9.0, 1.3), (13.0, 1.2), (19.5, 2.0)]
    w = [rng.uniform(0.5, 1.5) for _ in peaks]
    night_floor = 0.25 if rng.random() < 0.12 else 0.02  # ~12% of users are night owls
    prof = []
    for h in range(24):
        v = night_floor
        for (mu, sd), wi in zip(peaks, w, strict=True):
            v += wi * math.exp(-0.5 * ((h + 0.5 - mu) / sd) ** 2)
        prof.append(v)
    s = sum(prof)
    return [p / s for p in prof]


def build_world(
    n_users: int,
    n_merchants: int,
    days: int,
    start_ts: float,
    seed: int,
) -> World:
    rng = random.Random(seed)
    cities = INDIAN_CITIES
    city_weights = [c[3] for c in cities]

    # ---- merchants ----------------------------------------------------------
    merchants: dict[str, Merchant] = {}
    merchants_by_city: dict[str, list[str]] = {c[0]: [] for c in cities}
    online: list[str] = []
    cat_names = list(CATEGORIES)
    cat_weights = [CATEGORIES[c][3] for c in cat_names]
    for i in range(n_merchants):
        cat = rng.choices(cat_names, weights=cat_weights)[0]
        base, risk, p_online, _ = CATEGORIES[cat]
        is_online = rng.random() < p_online
        mid = f"M{i:05d}"
        created = start_ts - rng.uniform(30, 2000) * 86400
        if is_online:
            m = Merchant(mid, cat, "ONLINE", 0.0, 0.0, True, created, risk)
            online.append(mid)
        else:
            city, lat, lon, _ = rng.choices(cities, weights=city_weights)[0]
            m = Merchant(
                mid,
                cat,
                city,
                lat + rng.uniform(-0.05, 0.05),
                lon + rng.uniform(-0.05, 0.05),
                False,
                created,
                risk,
            )
            merchants_by_city[city].append(mid)
        merchants[mid] = m

    # a few card-present merchants abroad, used by the geo-jump pattern
    foreign_by_city: dict[str, list[str]] = {}
    fi = 0
    for city, lat, lon, country in FOREIGN_CITIES:
        foreign_by_city[city] = []
        for cat in ["electronics", "apparel", "jewellery", "forex", "travel"]:
            mid = f"MF{fi:04d}"
            fi += 1
            merchants[mid] = Merchant(
                mid, cat, city, lat, lon, False, start_ts - 400 * 86400, CATEGORIES[cat][1], country
            )
            foreign_by_city[city].append(mid)

    # ---- users ----------------------------------------------------------------
    users: list[User] = []
    for i in range(n_users):
        city, lat, lon, _ = rng.choices(cities, weights=city_weights)[0]
        n_dev = 1 if rng.random() < 0.6 else 2
        devices = [f"D{i:06d}{chr(97 + k)}" for k in range(n_dev)]
        # ~6% of users sometimes pay from a family member's phone (a device shared by 2 users)
        if i > 0 and rng.random() < 0.06:
            devices.append(users[rng.randrange(i)].devices[0])
        local_pool = merchants_by_city[city]
        n_fav = rng.randint(8, 20)
        favs: list[str] = []
        for _ in range(n_fav):
            pool = local_pool if (rng.random() < 0.55 and local_pool) else online
            favs.append(rng.choice(pool))
        upi = rng.uniform(0.4, 0.85)
        card = rng.uniform(0.05, 0.9) * (1 - upi)
        net = max(0.02, 1 - upi - card)
        users.append(
            User(
                user_id=f"U{i:06d}",
                home_city=city,
                home_lat=lat,
                home_lon=lon,
                account_created_ts=start_ts - rng.uniform(5, 1500) * 86400,
                spend_level=rng.lognormvariate(0, 0.45),
                daily_rate=max(0.05, rng.gammavariate(2.5, 0.14)),
                hour_weights=_hour_profile(rng),
                devices=devices,
                favourite_merchants=favs,
                channel_weights=(upi, card, net),
            )
        )

    ring_devices = [f"DRING{i:04d}" for i in range(max(10, n_users // 150))]
    return World(
        start_ts=start_ts,
        days=days,
        users=users,
        merchants=merchants,
        merchants_by_city=merchants_by_city,
        foreign_merchants_by_city=foreign_by_city,
        online_merchants=online,
        ring_devices=ring_devices,
        rng=rng,
    )


def local_hour(ts: float) -> float:
    return ((ts + IST_OFFSET) % 86400) / 3600.0


def sample_time_in_day(rng: random.Random, day_start_ts: float, hour_weights: list[float]) -> float:
    """Pick a UTC timestamp within a day according to a local-hour profile."""
    h = rng.choices(range(24), weights=hour_weights)[0]
    local_seconds = h * 3600 + rng.uniform(0, 3600)
    return day_start_ts + local_seconds - IST_OFFSET


def draw_amount(
    rng: random.Random, category: str, spend_level: float, multiplier: float = 1.0
) -> float:
    base = CATEGORIES[category][0]
    amt = base * spend_level * multiplier * rng.lognormvariate(0, 0.55)
    amt = max(1.0, amt)
    if amt > 500 and rng.random() < 0.6:
        amt = round(amt)
    return round(amt, 2)
