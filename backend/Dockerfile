# syntax=docker/dockerfile:1.7
# ==============================================================================
# Shifa42 backend — FastAPI + LangGraph
#
# Multi-stage. The builder resolves dependencies with uv; the runtime carries only
# the virtualenv and application code.
#
# Note on image size: sentence-transformers pulls torch, so this image is large
# (~2.5 GB) even on CPU wheels. That is inherent to running embeddings locally
# rather than calling a hosted embedding API — the tradeoff the project chose in
# exchange for zero API cost and no corpus text leaving the host.
# ==============================================================================

FROM python:3.11-slim AS builder

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    UV_LINK_MODE=copy

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential curl libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app
COPY pyproject.toml ./
COPY app ./app

RUN uv venv /opt/venv \
    && VIRTUAL_ENV=/opt/venv uv pip install --no-cache .

# ------------------------------------------------------------------ runtime

FROM python:3.11-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH" \
    HF_HOME=/app/.cache/huggingface

# libgomp1 is required by xgboost; curl is used by the healthcheck.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

# Non-root. The app writes only to data/ and .cache/, both owned below.
RUN useradd --create-home --uid 10001 shifa
WORKDIR /app

COPY --from=builder /opt/venv /opt/venv
COPY --chown=shifa:shifa app ./app
COPY --chown=shifa:shifa scripts ./scripts
COPY --chown=shifa:shifa pyproject.toml ./

RUN mkdir -p data/raw data/processed data/artifacts .cache \
    && chown -R shifa:shifa /app

USER shifa
EXPOSE 8000

# Uses the app's own component-aware health endpoint rather than a bare TCP check,
# so a container with an empty index reports unhealthy instead of merely "up".
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
    CMD curl -fsS http://localhost:8000/api/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
