"""Model bundle loading, prediction and SHAP explanations."""

from __future__ import annotations

import json
import logging
import tempfile
import threading
import time
from pathlib import Path

import numpy as np
import xgboost as xgb

from radar.model.registry import ModelRegistry
from radar.schemas import Reason

log = logging.getLogger(__name__)


class ModelBundle:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.metadata = json.loads((path / "metadata.json").read_text())
        self.version: str = self.metadata["version"]
        self.features: list[str] = self.metadata["features"]
        self.threshold: float = float(self.metadata["threshold"])
        self.booster = xgb.Booster()
        self.booster.load_model(str(path / "model.json"))
        self._explainer = None
        self._explainer_lock = threading.Lock()

    # Lazy: building a TreeExplainer costs ~100ms and is only needed for flagged txns.
    @property
    def explainer(self):
        if self._explainer is None:
            with self._explainer_lock:
                if self._explainer is None:
                    import shap

                    self._explainer = shap.TreeExplainer(self.booster)
        return self._explainer

    def vector(self, features: dict[str, float]) -> np.ndarray:
        return np.array([[float(features.get(f, 0.0)) for f in self.features]], dtype=np.float32)

    def score(self, features: dict[str, float]) -> float:
        d = xgb.DMatrix(self.vector(features), feature_names=self.features)
        return float(self.booster.predict(d)[0])

    def score_batch(self, rows: list[dict[str, float]]) -> np.ndarray:
        x = np.array(
            [[float(r.get(f, 0.0)) for f in self.features] for r in rows], dtype=np.float32
        )
        return self.booster.predict(xgb.DMatrix(x, feature_names=self.features))

    def explain(self, features: dict[str, float], top_k: int = 3) -> list[Reason]:
        x = self.vector(features)
        contrib = np.asarray(self.explainer.shap_values(x))[0]
        order = np.argsort(-np.abs(contrib))[:top_k]
        return [
            Reason(
                feature=self.features[i],
                value=float(x[0, i]),
                contribution=float(contrib[i]),
                direction="raises" if contrib[i] > 0 else "lowers",
            )
            for i in order
        ]


class ChampionLoader:
    """Keeps the live champion in memory and hot-swaps it when the registry changes."""

    def __init__(
        self, registry_uri: str, refresh_seconds: int = 60, cache_dir: Path | None = None
    ) -> None:
        self.registry = ModelRegistry(registry_uri)
        self.refresh_seconds = refresh_seconds
        self.cache_dir = cache_dir or Path(tempfile.mkdtemp(prefix="radar-models-"))
        self._bundle: ModelBundle | None = None
        self._marker: str | None = None
        self._last_check = 0.0
        self._lock = threading.Lock()

    def get(self) -> ModelBundle:
        now = time.time()
        if self._bundle is None or now - self._last_check > self.refresh_seconds:
            with self._lock:
                self._maybe_reload()
                self._last_check = now
        assert self._bundle is not None
        return self._bundle

    def _maybe_reload(self) -> None:
        marker = self.registry.version_marker()
        if self._bundle is not None and marker == self._marker:
            return
        champ = self.registry.champion()
        if champ is None:
            raise RuntimeError(f"no champion.json in registry {self.registry.uri}")
        version = champ["version"]
        if self._bundle is not None and self._bundle.version == version:
            self._marker = marker
            return
        local = self.registry.fetch(version, self.cache_dir)
        self._bundle = ModelBundle(local)
        self._marker = marker
        log.info("loaded champion model %s (threshold %.4f)", version, self._bundle.threshold)
