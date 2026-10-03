"""Per-user and per-device rolling state, updated once per transaction.

State is deliberately small and bounded (recent window pruned to 24h and capped, sets
capped) so that a DynamoDB item stays far below the 400KB limit and a read+write per
transaction stays cheap.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

from radar.schemas import Transaction

RECENT_WINDOW_SECONDS = 24 * 3600
RECENT_CAP = 500
KNOWN_DEVICES_CAP = 50
KNOWN_MERCHANTS_CAP = 300
DEVICE_USERS_CAP = 200


@dataclass
class UserState:
    user_id: str
    txn_count: int = 0
    amount_mean: float = 0.0
    amount_m2: float = 0.0  # Welford running sum of squared deviations
    recent: list[list[float]] = field(default_factory=list)  # [[ts, amount], ...] within 24h
    known_devices: dict[str, float] = field(default_factory=dict)  # device_id -> first seen ts
    known_merchants: list[str] = field(default_factory=list)
    first_ts: float | None = None
    last_ts: float | None = None
    last_lat: float | None = None
    last_lon: float | None = None
    last_city: str | None = None

    # ---- derived -------------------------------------------------------------
    @property
    def amount_std(self) -> float:
        if self.txn_count < 2:
            return 0.0
        return math.sqrt(self.amount_m2 / (self.txn_count - 1))

    def recent_within(self, ts: float, seconds: float) -> list[list[float]]:
        cutoff = ts - seconds
        return [r for r in self.recent if r[0] > cutoff and r[0] <= ts]

    # ---- mutation ------------------------------------------------------------
    def update(self, txn: Transaction) -> None:
        self.txn_count += 1
        delta = txn.amount - self.amount_mean
        self.amount_mean += delta / self.txn_count
        self.amount_m2 += delta * (txn.amount - self.amount_mean)

        cutoff = txn.ts - RECENT_WINDOW_SECONDS
        self.recent = [r for r in self.recent if r[0] > cutoff]
        self.recent.append([txn.ts, txn.amount])
        if len(self.recent) > RECENT_CAP:
            self.recent = self.recent[-RECENT_CAP:]

        if txn.device_id not in self.known_devices:
            self.known_devices[txn.device_id] = txn.ts
            if len(self.known_devices) > KNOWN_DEVICES_CAP:  # drop the oldest
                oldest = min(self.known_devices, key=self.known_devices.get)
                del self.known_devices[oldest]
        if txn.merchant_id not in self.known_merchants:
            self.known_merchants.append(txn.merchant_id)
            self.known_merchants = self.known_merchants[-KNOWN_MERCHANTS_CAP:]

        if self.first_ts is None:
            self.first_ts = txn.ts
        self.last_ts = txn.ts
        self.last_lat = txn.lat
        self.last_lon = txn.lon
        self.last_city = txn.city

    # ---- (de)serialisation ------------------------------------------------
    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> UserState:
        return cls(**d)


@dataclass
class DeviceState:
    device_id: str
    users: list[str] = field(default_factory=list)
    first_seen_ts: float | None = None
    txn_count: int = 0

    def update(self, txn: Transaction) -> None:
        self.txn_count += 1
        if txn.user_id not in self.users:
            self.users.append(txn.user_id)
            self.users = self.users[-DEVICE_USERS_CAP:]
        if self.first_seen_ts is None:
            self.first_seen_ts = txn.ts

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> DeviceState:
        return cls(**d)
