"""Retrieval Planner — the Adaptive RAG router.

This node is what makes the agentic path defensible rather than decorative. Running
every query through graph traversal, patient-context lookup and a multi-retry critic
loop would improve almost nothing on simple factual lookups while multiplying latency
and free-tier token spend. The planner decides which machinery a query actually
needs, and because the decision is logged, the eval harness can compare accuracy per
route against cost per route — which is the only way to show the complexity earns
its keep instead of asserting it.

The classification runs on the *fast* model tier deliberately. A router that costs
as much as the work it routes is not a saving.

On low confidence we widen rather than narrow: an uncertain router should not be the
thing that silently starves a hard question of evidence.
"""

from __future__ import annotations

import re
import time
from typing import Any

from app.agent.state import AgentState, trace_event
from app.llm import router as llm
from app.llm.prompts import PLANNER_SYSTEM
from app.logging_conf import get_logger

log = get_logger("agent.planner")

VALID_ROUTES = ("simple_factual", "needs_patient_context", "needs_relationship_reasoning")

# Cheap signals checked before the LLM. When these fire the route is unambiguous and
# an LLM call would only add latency and a chance of being wrong.
_RELATIONAL = re.compile(
    r"\b(interact|interaction|contraindicat|combin\w+ with|together with|instead of|"
    r"versus|vs\.?|compared? (?:to|with)|safer|alternative to|because of (?:his|her|their)|"
    r"given (?:his|her|their|the) \w+ (?:impairment|failure|disease))\b",
    re.I,
)
_PATIENT = re.compile(
    r"\b(this patient|the patient|patient \d+|my patient|their vitals|his vitals|her vitals|"
    r"triage|acuity|admitted|presenting with)\b",
    re.I,
)


def heuristic_route(query: str, has_patient_id: bool) -> str | None:
    if has_patient_id:
        return "needs_patient_context"
    if _RELATIONAL.search(query):
        return "needs_relationship_reasoning"
    if _PATIENT.search(query):
        return "needs_patient_context"
    return None


async def planner_node(state: AgentState) -> dict[str, Any]:
    t0 = time.perf_counter()
    query = state.get("normalized_query") or state["query"]
    has_patient = bool(state.get("patient_id"))

    heuristic = heuristic_route(query, has_patient)
    if heuristic:
        dt = (time.perf_counter() - t0) * 1000
        return {
            "route": heuristic,
            "route_confidence": 0.9,
            "route_reasoning": "matched a deterministic routing rule",
            "trace": [
                trace_event(
                    "planner", "routed", dt,
                    route=heuristic, confidence=0.9, method="heuristic",
                    reasoning="matched a deterministic routing rule",
                )
            ],
        }

    result = await llm.complete_json(
        [
            {"role": "system", "content": PLANNER_SYSTEM},
            {"role": "user", "content": f"Query: {query}"},
        ],
        tier=llm.Tier.FAST,
        default={"route": "simple_factual", "confidence": 0.4, "reasoning": "router fallback"},
    )

    route = str(result.get("route") or "simple_factual")
    if route not in VALID_ROUTES:
        log.warning("planner.invalid_route", got=route)
        route = "simple_factual"

    try:
        confidence = max(0.0, min(1.0, float(result.get("confidence", 0.5))))
    except (TypeError, ValueError):
        confidence = 0.5

    reasoning = str(result.get("reasoning") or "")[:240]

    # Widen on uncertainty. A router that is unsure is a poor reason to withhold the
    # graph retriever — the extra hop costs latency, while starving a hard question
    # of evidence costs an escalation or, worse, a thinly-grounded answer.
    if confidence < 0.5 and route == "simple_factual":
        route = "needs_relationship_reasoning"
        reasoning = f"{reasoning} | widened: low router confidence ({confidence:.2f})"

    dt = (time.perf_counter() - t0) * 1000
    log.info("planner.routed", run_id=state["run_id"], route=route, confidence=confidence)
    return {
        "route": route,
        "route_confidence": confidence,
        "route_reasoning": reasoning,
        "trace": [
            trace_event(
                "planner", "routed", dt,
                route=route, confidence=round(confidence, 3),
                method="llm", reasoning=reasoning,
            )
        ],
    }


def route_selector(state: AgentState) -> list[str]:
    """Fan-out edge: which retriever nodes run for this route.

    Returned as a list so LangGraph executes the selected retrievers concurrently —
    on the relationship route, the vector and graph retrievers have no dependency on
    each other and serializing them would double the retrieval latency for nothing.
    """
    route = state.get("route", "simple_factual")
    if route == "needs_relationship_reasoning":
        return ["vector_retriever", "graph_retriever"]
    if route == "needs_patient_context":
        return ["vector_retriever", "risk_node"]
    return ["vector_retriever"]
