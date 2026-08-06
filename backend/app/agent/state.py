"""Agent graph state.

`trace` is the governance backbone: every node appends one entry, and the API
streams those entries to the UI as they land. It is the same data the audit table
persists, so what the user watches in the "live trace" strip and what a reviewer
reads back months later are the same record — not a pretty summary of it.
"""

from __future__ import annotations

import operator
import time
from typing import Annotated, Any, Literal, TypedDict

from app.retrieval.types import SearchHit

Route = Literal["simple_factual", "needs_patient_context", "needs_relationship_reasoning"]
Outcome = Literal["answered", "escalated", "pending"]


class TraceEvent(TypedDict, total=False):
    node: str
    event: str
    ts: float
    duration_ms: float
    detail: dict[str, Any]


def trace_event(node: str, event: str, duration_ms: float = 0.0, **detail: Any) -> TraceEvent:
    return {
        "node": node,
        "event": event,
        "ts": time.time(),
        "duration_ms": round(duration_ms, 2),
        "detail": detail,
    }


class AgentState(TypedDict, total=False):
    # -- inputs
    run_id: str
    query: str
    patient_id: str | None

    # -- intake
    normalized_query: str
    entities: list[str]
    question_type: str
    contains_identifiers: bool

    # -- planner
    route: Route
    route_confidence: float
    route_reasoning: str

    # -- retrieval
    search_query: str
    dense_hits: list[SearchHit]
    sparse_hits: list[SearchHit]
    graph_hits: list[SearchHit]
    graph_paths: list[dict[str, Any]]
    fused_hits: list[SearchHit]
    context_blocks: list[dict[str, Any]]

    # -- risk
    risk: dict[str, Any] | None

    # -- generation
    draft: str
    attempt: int
    reformulation_strategy: str

    # -- critic
    groundedness: float
    critic_report: dict[str, Any]
    best_groundedness: float

    # -- outcome
    answer: str
    citations: list[dict[str, Any]]
    outcome: Outcome
    escalation_reason: str

    # -- meta
    degraded: bool
    trace: Annotated[list[TraceEvent], operator.add]


def initial_state(query: str, run_id: str, patient_id: str | None = None) -> AgentState:
    return AgentState(
        run_id=run_id,
        query=query,
        patient_id=patient_id,
        normalized_query=query,
        search_query=query,
        entities=[],
        dense_hits=[],
        sparse_hits=[],
        graph_hits=[],
        graph_paths=[],
        fused_hits=[],
        context_blocks=[],
        risk=None,
        draft="",
        attempt=0,
        groundedness=0.0,
        best_groundedness=0.0,
        critic_report={},
        answer="",
        citations=[],
        outcome="pending",
        escalation_reason="",
        degraded=False,
        trace=[],
    )
