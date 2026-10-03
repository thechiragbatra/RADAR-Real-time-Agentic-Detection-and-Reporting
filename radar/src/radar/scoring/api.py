"""Scoring service.

    POST /score           score one feature vector (what the stream consumer calls)
    POST /score/batch     score many (used by the drift monitor and load tests)
    GET  /health          liveness + current model version
    GET  /model           champion metadata (threshold, metrics, training window)
    GET  /metrics         Prometheus text format: request count, latency histogram, flag rate

Runs as a container on ECS Fargate behind an ALB; see infra/terraform/ecs.tf.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse

from radar.config import settings
from radar.schemas import ScoreRequest, ScoreResponse
from radar.scoring.scorer import ChampionLoader

log = logging.getLogger("radar.scoring")

LATENCY_BUCKETS_MS = [1, 2, 5, 10, 20, 50, 100, 250, 500, 1000]


class Metrics:
    """Tiny in-process metrics so we do not need a Prometheus client dependency."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.requests = 0
        self.flagged = 0
        self.errors = 0
        self.latency_sum_ms = 0.0
        self.buckets = dict.fromkeys(LATENCY_BUCKETS_MS, 0)

    def observe(self, latency_ms: float, flagged: bool) -> None:
        with self.lock:
            self.requests += 1
            self.flagged += int(flagged)
            self.latency_sum_ms += latency_ms
            for b in LATENCY_BUCKETS_MS:
                if latency_ms <= b:
                    self.buckets[b] += 1

    def render(self, version: str) -> str:
        lines = [
            "# TYPE radar_score_requests_total counter",
            f"radar_score_requests_total {self.requests}",
            "# TYPE radar_score_flagged_total counter",
            f"radar_score_flagged_total {self.flagged}",
            "# TYPE radar_score_errors_total counter",
            f"radar_score_errors_total {self.errors}",
            "# TYPE radar_score_latency_ms histogram",
        ]
        for b in LATENCY_BUCKETS_MS:
            lines.append(f'radar_score_latency_ms_bucket{{le="{b}"}} {self.buckets[b]}')
        lines.append(f'radar_score_latency_ms_bucket{{le="+Inf"}} {self.requests}')
        lines.append(f"radar_score_latency_ms_sum {self.latency_sum_ms:.3f}")
        lines.append(f"radar_score_latency_ms_count {self.requests}")
        lines.append(f'radar_model_info{{version="{version}"}} 1')
        return "\n".join(lines) + "\n"


def create_app(registry_uri: str | None = None) -> FastAPI:
    loader = ChampionLoader(registry_uri or settings.model_uri, settings.model_refresh_seconds)
    metrics = Metrics()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        loader.get()  # fail fast if there is no champion
        yield

    app = FastAPI(title="RADAR scoring service", version="0.1.0", lifespan=lifespan)
    app.state.loader = loader
    app.state.metrics = metrics

    @app.get("/health")
    def health() -> dict:
        b = loader.get()
        return {"status": "ok", "model_version": b.version, "threshold": b.threshold}

    @app.get("/model")
    def model() -> dict:
        b = loader.get()
        return {k: v for k, v in b.metadata.items() if k not in ("feature_stats",)}

    @app.get("/metrics", response_class=PlainTextResponse)
    def prom() -> str:
        return metrics.render(loader.get().version)

    @app.post("/score", response_model=ScoreResponse)
    def score(req: ScoreRequest) -> ScoreResponse:
        t0 = time.perf_counter()
        b = loader.get()
        try:
            s = b.score(req.features)
        except Exception as exc:  # pragma: no cover - defensive
            metrics.errors += 1
            log.exception("scoring failed for %s", req.txn_id)
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        flagged = s >= b.threshold
        explain = (
            req.explain
            if req.explain is not None
            else (flagged or not settings.explain_only_flagged)
        )
        reasons = b.explain(req.features) if explain else []
        latency_ms = (time.perf_counter() - t0) * 1000
        metrics.observe(latency_ms, flagged)
        return ScoreResponse(
            txn_id=req.txn_id,
            score=s,
            flagged=flagged,
            threshold=b.threshold,
            model_version=b.version,
            reasons=reasons,
            latency_ms=round(latency_ms, 3),
        )

    @app.post("/score/batch")
    def score_batch(reqs: list[ScoreRequest]) -> list[dict]:
        b = loader.get()
        scores = b.score_batch([r.features for r in reqs])
        return [
            {"txn_id": r.txn_id, "score": float(s), "flagged": bool(s >= b.threshold)}
            for r, s in zip(reqs, scores, strict=True)
        ]

    return app


app = create_app()


def main() -> None:
    import uvicorn

    uvicorn.run(
        "radar.scoring.api:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", "8000")),
        workers=int(os.environ.get("WEB_CONCURRENCY", "2")),
        log_level=settings.log_level.lower(),
    )


if __name__ == "__main__":
    main()
