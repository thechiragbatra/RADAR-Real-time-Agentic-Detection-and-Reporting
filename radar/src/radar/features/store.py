"""Feature store: where rolling user/device state lives.

``InMemoryFeatureStore`` is used for training replays and tests. ``DynamoDBFeatureStore``
is the production store: one item per user and per device, state stored as a JSON
document, read once and written once per transaction.

Why no locking? The Kinesis partition key is ``user_id`` and a Lambda processes one
shard's batch sequentially, so two updates for the same user never run concurrently.
Device state can in theory be touched from two shards at once; we accept the rare
lost-update there because it only affects a count used as a soft signal.
"""

from __future__ import annotations

import json
import time
from typing import Protocol

from radar.features.state import DeviceState, UserState


class FeatureStore(Protocol):
    def get_user_state(self, user_id: str) -> UserState: ...
    def put_user_state(self, state: UserState) -> None: ...
    def get_device_state(self, device_id: str) -> DeviceState: ...
    def put_device_state(self, state: DeviceState) -> None: ...


class InMemoryFeatureStore:
    def __init__(self) -> None:
        self.users: dict[str, UserState] = {}
        self.devices: dict[str, DeviceState] = {}

    def get_user_state(self, user_id: str) -> UserState:
        return self.users.get(user_id) or UserState(user_id=user_id)

    def put_user_state(self, state: UserState) -> None:
        self.users[state.user_id] = state

    def get_device_state(self, device_id: str) -> DeviceState:
        return self.devices.get(device_id) or DeviceState(device_id=device_id)

    def put_device_state(self, state: DeviceState) -> None:
        self.devices[state.device_id] = state


class DynamoDBFeatureStore:
    """Single table, key ``pk`` = ``USER#<id>`` or ``DEVICE#<id>``; state in a JSON string."""

    def __init__(self, table_name: str, region: str | None = None, resource=None) -> None:
        if resource is None:
            import boto3

            resource = boto3.resource("dynamodb", region_name=region)
        self.table = resource.Table(table_name)

    def _get(self, pk: str) -> dict | None:
        resp = self.table.get_item(Key={"pk": pk}, ConsistentRead=True)
        item = resp.get("Item")
        return json.loads(item["state"]) if item else None

    def _put(self, pk: str, state: dict) -> None:
        self.table.put_item(
            Item={"pk": pk, "state": json.dumps(state), "updated_at": int(time.time())}
        )

    def get_user_state(self, user_id: str) -> UserState:
        d = self._get(f"USER#{user_id}")
        return UserState.from_dict(d) if d else UserState(user_id=user_id)

    def put_user_state(self, state: UserState) -> None:
        self._put(f"USER#{state.user_id}", state.to_dict())

    def get_device_state(self, device_id: str) -> DeviceState:
        d = self._get(f"DEVICE#{device_id}")
        return DeviceState.from_dict(d) if d else DeviceState(device_id=device_id)

    def put_device_state(self, state: DeviceState) -> None:
        self._put(f"DEVICE#{state.device_id}", state.to_dict())
