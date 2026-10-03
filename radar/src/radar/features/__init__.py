"""Stateful feature engineering.

The same code path produces features for offline training (replaying history through an
in-memory store) and for online scoring (DynamoDB store), which removes train/serve skew
by construction. See ``engineering.py`` for the feature definitions.
"""

from radar.features.engineering import (
    FEATURE_NAMES,
    build_training_frame,
    compute_features,
    featurize_and_update,
)
from radar.features.reference import LocalReferenceData, ReferenceData
from radar.features.state import DeviceState, UserState
from radar.features.store import DynamoDBFeatureStore, FeatureStore, InMemoryFeatureStore

__all__ = [
    "FEATURE_NAMES",
    "DeviceState",
    "DynamoDBFeatureStore",
    "FeatureStore",
    "InMemoryFeatureStore",
    "LocalReferenceData",
    "ReferenceData",
    "UserState",
    "build_training_frame",
    "compute_features",
    "featurize_and_update",
]
