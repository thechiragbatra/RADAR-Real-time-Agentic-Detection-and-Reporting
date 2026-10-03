# Load test

```
make serve                                  # terminal 1
make loadtest ALB=http://localhost:8000     # terminal 2 (50 users, 2 min)
```

Against AWS, pass the ALB URL and the header key:

```
RADAR_SCORER_API_KEY=... locust -f loadtest/locustfile.py --host http://<alb-dns> --headless -u 100 -r 20 -t 5m --csv loadtest/results/aws
```

## Reference run (local, 2 uvicorn workers, Locust on the same 2-vCPU container)

| | requests | req/s | p50 | p95 | p99 | failures |
|---|---|---|---|---|---|---|
| POST /score | 42,424 | 717 | 27 ms | 63 ms | 84 ms | 0.04% (server-side > 100 ms budget) |

Server-side latency reported by the service itself (`latency_ms` in the response) is
0.4 ms median for an unflagged transaction and ~1.5 ms for a flagged one (SHAP for the
top-3 reasons). The round-trip numbers above are dominated by the client and the shared
CPU; on Fargate behind an ALB expect ~5-15 ms p50 from a Lambda in the same region.

What to look for on AWS:

* ALB `TargetResponseTime` p95 stays under the 100 ms alarm threshold
* ECS desired count climbs from 1 to 3 within ~3 minutes (target: 300 req/min per task)
* `radar_score_latency_ms` histogram on `/metrics` - the service's own view
