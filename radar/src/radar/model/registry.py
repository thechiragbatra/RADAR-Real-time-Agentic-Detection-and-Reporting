"""A minimal model registry on a local directory or an S3 prefix.

Layout::

    <root>/
      champion.json                 {"version": "...", "promoted_at": ..., "history": [...]}
      <version>/model.json          XGBoost booster
      <version>/metadata.json       features, threshold, metrics, training window
      <version>/reference.json      PSI reference bins
      <version>/shap_summary.json   global feature importance
      <version>/case_bank.parquet   labelled examples for the agent's similar-case search

Promotion is a single-object write of ``champion.json``; the scorer polls its ETag /
mtime and hot-swaps the model without a redeploy. MLflow is used for experiment tracking
when installed (see ``train.py``) but is not on the serving path.
"""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

BUNDLE_FILES = [
    "model.json",
    "metadata.json",
    "reference.json",
    "shap_summary.json",
    "case_bank.parquet",
]


class ModelRegistry:
    def __init__(self, uri: str, region: str | None = None) -> None:
        self.uri = uri.rstrip("/")
        self.is_s3 = self.uri.startswith("s3://")
        if self.is_s3:
            import boto3

            self._s3 = boto3.client("s3", region_name=region)
            without = self.uri[len("s3://") :]
            self.bucket, _, self.prefix = without.partition("/")
        else:
            self.root = Path(self.uri)
            self.root.mkdir(parents=True, exist_ok=True)

    # ---- low-level -----------------------------------------------------------
    def _key(self, *parts: str) -> str:
        return "/".join(p for p in (self.prefix, *parts) if p)

    def read_text(self, *parts: str) -> str | None:
        if self.is_s3:
            try:
                obj = self._s3.get_object(Bucket=self.bucket, Key=self._key(*parts))
            except self._s3.exceptions.NoSuchKey:
                return None
            return obj["Body"].read().decode()
        p = self.root.joinpath(*parts)
        return p.read_text() if p.exists() else None

    def write_text(self, text: str, *parts: str) -> None:
        if self.is_s3:
            self._s3.put_object(Bucket=self.bucket, Key=self._key(*parts), Body=text.encode())
        else:
            p = self.root.joinpath(*parts)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(text)

    def version_marker(self) -> str | None:
        """Cheap change detector for the scorer's refresh loop (ETag on S3, mtime locally)."""
        if self.is_s3:
            try:
                head = self._s3.head_object(Bucket=self.bucket, Key=self._key("champion.json"))
                return head["ETag"]
            except Exception:
                return None
        p = self.root / "champion.json"
        return str(p.stat().st_mtime_ns) if p.exists() else None

    # ---- registry operations ------------------------------------------------
    def champion(self) -> dict | None:
        text = self.read_text("champion.json")
        return json.loads(text) if text else None

    def list_versions(self) -> list[str]:
        if self.is_s3:
            resp = self._s3.list_objects_v2(
                Bucket=self.bucket, Prefix=self._key("") + "/" if self.prefix else "", Delimiter="/"
            )
            return sorted(
                p["Prefix"].rstrip("/").split("/")[-1] for p in resp.get("CommonPrefixes", [])
            )
        return sorted(p.name for p in self.root.iterdir() if p.is_dir())

    def publish(self, local_dir: Path, version: str) -> None:
        for name in BUNDLE_FILES:
            src = local_dir / name
            if not src.exists():
                continue
            if self.is_s3:
                self._s3.upload_file(str(src), self.bucket, self._key(version, name))
            else:
                dst = self.root / version / name
                if src.resolve() != dst.resolve():
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src, dst)

    def promote(self, version: str, reason: str, metrics: dict | None = None) -> dict:
        current = self.champion() or {"history": []}
        entry = {
            "version": version,
            "promoted_at": time.time(),
            "reason": reason,
            "metrics": metrics or {},
        }
        doc = {
            **entry,
            "previous": current.get("version"),
            "history": (current.get("history", []) + [entry])[-50:],
        }
        self.write_text(json.dumps(doc, indent=2), "champion.json")
        return doc

    def fetch(self, version: str, dest: Path) -> Path:
        """Materialise a version locally (no-op copy for local registries)."""
        out = dest / version
        out.mkdir(parents=True, exist_ok=True)
        for name in BUNDLE_FILES:
            if self.is_s3:
                try:
                    self._s3.download_file(self.bucket, self._key(version, name), str(out / name))
                except Exception:
                    pass
            else:
                src = self.root / version / name
                if src.exists() and src.resolve() != (out / name).resolve():
                    shutil.copy2(src, out / name)
        return out

    def metadata(self, version: str) -> dict | None:
        text = self.read_text(version, "metadata.json")
        return json.loads(text) if text else None
