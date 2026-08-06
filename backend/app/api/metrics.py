"""GET /metrics — Prometheus exposition format.

Real metric families, not a decorative endpoint: counters for runs and outcomes,
a histogram for latency, and gauges for corpus size and LLM spend. Anything a
Prometheus server could scrape and alert on.

The one worth alerting on is `shifa42_escalations_total / shifa42_runs_total`. A
sudden climb means retrieval has degraded — a stale index, a provider returning
junk — and it surfaces as a metric before any user reports a bad answer. That is
the practical argument for instrumenting this system at all.
"""

from __future__ import annotations

from fastapi import APIRouter, Response
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)

from app.config import settings
from app.logging_conf import get_logger

log = get_logger("api.metrics")
router = APIRouter()

REGISTRY = CollectorRegistry()

RUNS = Counter("shifa42_runs_total", "Agent runs started", registry=REGISTRY)
OUTCOMES = Counter(
    "shifa42_outcomes_total", "Agent run outcomes", ["outcome"], registry=REGISTRY
)
ROUTES = Counter(
    "shifa42_routes_total", "Retrieval routes chosen", ["route"], registry=REGISTRY
)
ESCALATIONS = Counter("shifa42_escalations_total", "Runs escalated to a physician", registry=REGISTRY)
RETRIES = Counter("shifa42_reformulations_total", "Query reformulations", registry=REGISTRY)

LATENCY = Histogram(
    "shifa42_run_duration_seconds",
    "End-to-end agent run duration",
    buckets=(0.5, 1, 2, 3, 5, 8, 13, 21, 34, 60),
    registry=REGISTRY,
)
GROUNDEDNESS = Histogram(
    "shifa42_groundedness",
    "Groundedness score of final answers",
    buckets=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.9, 1.0),
    registry=REGISTRY,
)

CHUNKS_INDEXED = Gauge("shifa42_chunks_indexed", "Child chunks in the vector index", registry=REGISTRY)
GRAPH_NODES = Gauge("shifa42_graph_nodes", "Knowledge graph nodes", registry=REGISTRY)
GRAPH_EDGES = Gauge("shifa42_graph_edges", "Knowledge graph edges", registry=REGISTRY)
LLM_CALLS = Gauge("shifa42_llm_calls_total", "LLM calls made", registry=REGISTRY)
LLM_TOKENS = Gauge("shifa42_llm_tokens_total", "LLM tokens consumed", ["kind"], registry=REGISTRY)
DEGRADED = Gauge("shifa42_degraded_mode", "1 when running without a real LLM provider", registry=REGISTRY)


def record_run(result: dict) -> None:
    """Called after each agent run. Never raises into the request path."""
    try:
        RUNS.inc()
        outcome = result.get("outcome", "unknown")
        OUTCOMES.labels(outcome=outcome).inc()
        if result.get("route"):
            ROUTES.labels(route=result["route"]).inc()
        if outcome == "escalated":
            ESCALATIONS.inc()
        attempts = int(result.get("attempts", 1) or 1)
        if attempts > 1:
            RETRIES.inc(attempts - 1)
        if result.get("groundedness") is not None:
            GROUNDEDNESS.observe(float(result["groundedness"]))
        trace = result.get("trace") or []
        total_ms = sum(float(e.get("duration_ms", 0) or 0) for e in trace)
        LATENCY.observe(total_ms / 1000.0)
    except Exception as exc:  # noqa: BLE001
        log.warning("metrics.record_failed", error=str(exc)[:160])


def _refresh_gauges() -> None:
    from app.llm.router import IS_DEGRADED, USAGE

    DEGRADED.set(1 if IS_DEGRADED else 0)
    snap = USAGE.snapshot()
    LLM_CALLS.set(snap["calls"])
    LLM_TOKENS.labels(kind="prompt").set(snap["prompt_tokens"])
    LLM_TOKENS.labels(kind="completion").set(snap["completion_tokens"])

    try:
        from app.retrieval.store import get_store

        CHUNKS_INDEXED.set(get_store().count())
    except Exception:  # noqa: BLE001
        CHUNKS_INDEXED.set(0)

    try:
        from app.retrieval.graph_store import get_graph

        kg = get_graph()
        if kg:
            s = kg.stats()
            GRAPH_NODES.set(s["nodes"])
            GRAPH_EDGES.set(s["edges"])
    except Exception:  # noqa: BLE001
        pass


@router.get("/metrics")
async def metrics() -> Response:
    _refresh_gauges()
    return Response(content=generate_latest(REGISTRY), media_type=CONTENT_TYPE_LATEST)


@router.get("/metrics/summary")
async def metrics_summary() -> dict:
    """JSON view of the same counters, for the dashboard's animated stat tiles.

    Prometheus text is awkward to parse in the browser, and the UI needs these
    numbers to be real rather than placeholder copy.
    """
    from app.audit.store import audit
    from app.llm.router import IS_DEGRADED, USAGE, provider_info

    stats = audit.stats()

    chunks = 0
    try:
        from app.retrieval.store import get_store

        chunks = get_store().count()
    except Exception:  # noqa: BLE001
        pass

    graph_stats = {}
    try:
        from app.retrieval.graph_store import get_graph

        kg = get_graph()
        graph_stats = kg.stats() if kg else {}
    except Exception:  # noqa: BLE001
        pass

    return {
        "chunks_indexed": chunks,
        "graph": graph_stats,
        "runs": stats,
        "llm": USAGE.snapshot(),
        "degraded": IS_DEGRADED,
        "provider": provider_info(),
        "thresholds": {
            "groundedness": settings.groundedness_threshold,
            "max_reformulations": settings.max_reformulations,
        },
    }
