"""Pydantic v2 request/response models.

Every endpoint validates its input. For a system whose selling point is that it
knows what it does and does not know, accepting unvalidated input would be an odd
place to start being relaxed.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


class QueryRequest(BaseModel):
    query: str = Field(min_length=3, max_length=2000, description="The clinical question.")
    patient_id: str | None = Field(
        default=None, max_length=64,
        description="Optional MIMIC-IV-ED Demo stay_id or subject_id for patient context.",
    )
    stream: bool = Field(default=True, description="Stream the agent trace over SSE.")

    @field_validator("query")
    @classmethod
    def _strip(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("query must not be blank")
        return v

    @field_validator("patient_id")
    @classmethod
    def _clean_pid(cls, v: str | None) -> str | None:
        if v is None:
            return None
        v = v.strip()
        if not v:
            return None
        if not v.isalnum():
            raise ValueError("patient_id must be alphanumeric")
        return v


class Citation(BaseModel):
    index: int
    title: str
    source: str = ""
    section: str = ""
    url: str = ""
    score: float = 0.0
    retriever: str = ""
    components: dict[str, float] = Field(default_factory=dict)
    snippet: str = ""


class TraceEntry(BaseModel):
    node: str
    event: str
    ts: float = 0.0
    duration_ms: float = 0.0
    detail: dict[str, Any] = Field(default_factory=dict)


class QueryResponse(BaseModel):
    run_id: str
    query: str
    answer: str
    outcome: Literal["answered", "escalated", "pending"]
    escalated: bool
    escalation_reason: str = ""
    citations: list[Citation] = Field(default_factory=list)
    groundedness: float = 0.0
    best_groundedness: float = 0.0
    critic_report: dict[str, Any] = Field(default_factory=dict)
    route: str = ""
    route_confidence: float = 0.0
    route_reasoning: str = ""
    entities: list[str] = Field(default_factory=list)
    attempts: int = 0
    risk: dict[str, Any] | None = None
    graph_paths: list[dict[str, Any]] = Field(default_factory=list)
    sources_reviewed: int = 0
    contains_identifiers: bool = False
    degraded: bool = False
    trace: list[TraceEntry] = Field(default_factory=list)
    llm_usage: dict[str, Any] = Field(default_factory=dict)
    provider: dict[str, Any] = Field(default_factory=dict)


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    version: str
    components: dict[str, Any]
    disclaimer: str


class RiskResponse(BaseModel):
    patient_id: str
    risk_score: float
    band: str
    outcome_predicted: str
    model: str
    matched_on: str = ""
    top_features: list[str] = Field(default_factory=list)
    observed_triage: dict[str, Any] = Field(default_factory=dict)
    model_performance: dict[str, Any] = Field(default_factory=dict)
    caveat: str = ""
