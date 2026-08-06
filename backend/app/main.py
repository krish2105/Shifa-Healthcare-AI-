"""FastAPI application entrypoint.

Security baseline: CORS locked to an explicit allow-list (never "*" — this API
streams clinical content), rate limiting via slowapi, Pydantic v2 validation on
every endpoint, and no PHI persisted anywhere.

The vector store and embedding model are warmed at startup rather than on first
request. Loading bge-large costs ~1.3 GB and several seconds, and paying that on a
user's first query looks exactly like a hang.
"""

from __future__ import annotations

import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from app.api import health as health_api
from app.api import metrics as metrics_api
from app.api import query as query_api
from app.api import risk as risk_api
from app.config import settings
from app.logging_conf import configure_logging, get_logger

configure_logging()
log = get_logger("main")

limiter = Limiter(key_func=get_remote_address, default_limits=[settings.rate_limit])


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("startup.begin", env=settings.app_env)

    from app.llm.router import IS_DEGRADED, provider_info

    log.info("startup.llm", **provider_info())
    if IS_DEGRADED:
        log.warning(
            "startup.degraded",
            msg="No LLM credential configured — responses will be deterministic "
            "placeholders flagged degraded=true. Set GROQ_API_KEY in backend/.env.",
        )

    try:
        from app.retrieval.store import get_store

        store = get_store()
        log.info("startup.vector_store", **store.health())
    except Exception as exc:  # noqa: BLE001
        log.error("startup.vector_store_failed", error=str(exc)[:200])

    try:
        from app.retrieval.bm25 import get_bm25

        bm = get_bm25()
        log.info("startup.bm25", ready=bm is not None, chunks=bm.count() if bm else 0)
    except Exception as exc:  # noqa: BLE001
        log.error("startup.bm25_failed", error=str(exc)[:200])

    try:
        from app.retrieval.embedder import get_model

        get_model()  # warm the encoder so first query is not a cold load
    except Exception as exc:  # noqa: BLE001
        log.error("startup.embedder_failed", error=str(exc)[:200])

    log.info("startup.ready")
    yield
    log.info("shutdown")


app = FastAPI(
    title="Shifa42 — Agentic Clinical Intelligence Copilot",
    description=(
        "Clinical decision support grounded in published guideline literature, with an "
        "automated groundedness check before any answer ships and a physician-escalation "
        "path when the evidence is insufficient.\n\n"
        "**Research and educational demonstration. Not a certified medical device.**"
    ),
    version=health_api.VERSION,
    lifespan=lifespan,
    docs_url="/docs",
    openapi_url="/openapi.json",
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["X-Run-Id"],
)


@app.middleware("http")
async def access_log(request: Request, call_next):
    t0 = time.perf_counter()
    response = await call_next(request)
    dt = (time.perf_counter() - t0) * 1000
    if not request.url.path.startswith(("/metrics", "/api/health")):
        log.info(
            "http",
            method=request.method,
            path=request.url.path,
            status=response.status_code,
            duration_ms=round(dt, 1),
        )
    return response


@app.exception_handler(Exception)
async def unhandled(request: Request, exc: Exception):
    log.error("unhandled_exception", path=request.url.path, error=str(exc)[:400])
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal error. See server logs.", "path": request.url.path},
    )


app.include_router(health_api.router, prefix=settings.api_prefix, tags=["system"])
app.include_router(query_api.router, prefix=settings.api_prefix, tags=["agent"])
app.include_router(risk_api.router, prefix=settings.api_prefix, tags=["risk"])
app.include_router(metrics_api.router, tags=["observability"])


@app.get("/", include_in_schema=False)
async def root() -> dict:
    return {
        "service": "shifa42",
        "version": health_api.VERSION,
        "docs": "/docs",
        "health": f"{settings.api_prefix}/health",
        "disclaimer": health_api.DISCLAIMER,
    }
