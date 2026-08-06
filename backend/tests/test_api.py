"""API contract tests.

Run against the real app with TestClient, so route wiring, Pydantic validation and
the degraded-mode reporting are all exercised together.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_health_reports_every_component(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] in ("ok", "degraded")
    for component in ("llm", "vector_store", "bm25", "knowledge_graph", "risk_model"):
        assert component in body["components"]


def test_health_carries_the_disclaimer(client):
    """The disclaimer is part of the API contract, not only the UI."""
    body = client.get("/api/health").json()
    assert "not a certified medical device" in body["disclaimer"].lower()


def test_health_is_honest_about_degraded_mode(client):
    """A stack running on the stub provider must not report itself as healthy."""
    body = client.get("/api/health").json()
    if body["components"]["llm"]["degraded"]:
        assert body["status"] == "degraded"


def test_metrics_endpoint_is_prometheus_format(client):
    r = client.get("/metrics")
    assert r.status_code == 200
    text = r.text
    assert "# HELP shifa42_runs_total" in text
    assert "# TYPE" in text


def test_metrics_summary_is_json(client):
    body = client.get("/metrics/summary").json()
    for key in ("chunks_indexed", "runs", "llm", "degraded", "thresholds"):
        assert key in body


def test_query_rejects_blank_input(client):
    assert client.post("/api/query", json={"query": "   ", "stream": False}).status_code == 422


def test_query_rejects_short_input(client):
    assert client.post("/api/query", json={"query": "hi", "stream": False}).status_code == 422


def test_query_rejects_oversized_input(client):
    r = client.post("/api/query", json={"query": "x" * 5000, "stream": False})
    assert r.status_code == 422


def test_query_rejects_non_alphanumeric_patient_id(client):
    r = client.post(
        "/api/query",
        json={"query": "What is first-line therapy?", "patient_id": "../../etc/passwd", "stream": False},
    )
    assert r.status_code == 422


def test_risk_score_rejects_bad_patient_id(client):
    assert client.get("/api/risk-score/not%20valid!").status_code in (404, 422)


def test_risk_score_404s_for_unknown_patient(client):
    r = client.get("/api/risk-score/99999999")
    assert r.status_code in (404, 503)


def test_info_exposes_measured_manifests(client):
    body = client.get("/api/info").json()
    for key in ("corpus", "ed_cohort", "risk_model", "benchmarks", "config"):
        assert key in body


def test_audit_endpoint_returns_stats(client):
    body = client.get("/api/audit?limit=5").json()
    assert "entries" in body
    assert "stats" in body


def test_cors_is_not_wildcarded():
    """A wildcard CORS origin on an API that streams clinical content is a real
    finding, not a style preference."""
    from app.config import settings

    assert "*" not in settings.cors_origins
