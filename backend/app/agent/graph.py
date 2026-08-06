"""The LangGraph agent graph.

Topology matches the architecture diagram one-for-one:

    START → intake → planner ─┬→ vector_retriever ─┐
                              ├→ graph_retriever  ─┼→ fusion → draft → critic
                              └→ risk_node        ─┘                     │
                                                                         │
        final_composer → END  ←── pass ──────────────────────────────────┤
        escalate       → END  ←── fail, retries exhausted ───────────────┤
        reformulate ──→ (back to the retrievers) ←── fail, retries left ─┘

Two structural points.

**The retriever fan-out is genuinely parallel.** `route_selector` returns a list, so
LangGraph runs the selected retrievers concurrently and `fusion` joins them. On the
relationship route, vector search and graph traversal have no dependency on each
other; serializing them would double retrieval latency for nothing.

**The retry loop re-enters at retrieval, not at drafting.** A groundedness failure
means we found the wrong evidence. Re-drafting against the same context would only
produce a differently-worded ungrounded answer.
"""

from __future__ import annotations

import uuid
from typing import Any

from langgraph.graph import END, START, StateGraph

from app.agent.nodes.composer import draft_node, final_node
from app.agent.nodes.critic import critic_decision, critic_node, escalate_node, reformulate_node
from app.agent.nodes.fusion import fusion_node
from app.agent.nodes.intake import intake_node
from app.agent.nodes.planner import planner_node, route_selector
from app.agent.nodes.retrievers import graph_retriever_node, risk_node, vector_retriever_node
from app.agent.state import AgentState, initial_state
from app.logging_conf import get_logger

log = get_logger("agent.graph")

RETRIEVER_NODES = ["vector_retriever", "graph_retriever", "risk_node"]


def build_graph():
    g = StateGraph(AgentState)

    g.add_node("intake", intake_node)
    g.add_node("planner", planner_node)
    g.add_node("vector_retriever", vector_retriever_node)
    g.add_node("graph_retriever", graph_retriever_node)
    g.add_node("risk_node", risk_node)
    g.add_node("fusion", fusion_node)
    g.add_node("draft", draft_node)
    g.add_node("critic", critic_node)
    g.add_node("reformulate", reformulate_node)
    g.add_node("final_composer", final_node)
    g.add_node("escalate", escalate_node)

    g.add_edge(START, "intake")
    g.add_edge("intake", "planner")

    # Adaptive fan-out — only the retrievers this route needs.
    g.add_conditional_edges("planner", route_selector, RETRIEVER_NODES)

    for node in RETRIEVER_NODES:
        g.add_edge(node, "fusion")

    g.add_edge("fusion", "draft")
    g.add_edge("draft", "critic")

    g.add_conditional_edges(
        "critic",
        critic_decision,
        {"final_composer": "final_composer", "reformulate": "reformulate", "escalate": "escalate"},
    )

    # Retry re-enters retrieval with the rewritten query.
    g.add_conditional_edges("reformulate", route_selector, RETRIEVER_NODES)

    g.add_edge("final_composer", END)
    g.add_edge("escalate", END)

    return g.compile()


_compiled = None


def get_agent():
    global _compiled
    if _compiled is None:
        _compiled = build_graph()
        log.info("agent.compiled", nodes=11)
    return _compiled


async def run_agent(
    query: str, *, patient_id: str | None = None, run_id: str | None = None
) -> dict[str, Any]:
    """Run the graph to completion and return a serializable result."""
    rid = run_id or uuid.uuid4().hex[:16]
    agent = get_agent()

    final_state = await agent.ainvoke(
        initial_state(query, rid, patient_id),
        # Generous: the worst case is 3 drafts + 3 critics + retrieval, and hitting a
        # recursion cap mid-flight would look like a hang rather than an escalation.
        config={"recursion_limit": 60},
    )
    return serialize_result(final_state)


def serialize_result(state: AgentState) -> dict[str, Any]:
    """Flatten graph state into the API response shape.

    SearchHit objects are dropped — the API exposes `citations` (what the answer
    actually rests on) and `trace` (what happened), not the raw retrieval objects.
    """
    from app.llm.router import USAGE, provider_info

    return {
        "run_id": state.get("run_id", ""),
        "query": state.get("query", ""),
        "normalized_query": state.get("normalized_query", ""),
        "answer": state.get("answer", ""),
        "outcome": state.get("outcome", "pending"),
        "escalated": state.get("outcome") == "escalated",
        "escalation_reason": state.get("escalation_reason", ""),
        "citations": state.get("citations", []),
        "groundedness": round(state.get("groundedness", 0.0), 4),
        "best_groundedness": round(state.get("best_groundedness", 0.0), 4),
        "critic_report": state.get("critic_report", {}),
        "route": state.get("route", ""),
        "route_confidence": state.get("route_confidence", 0.0),
        "route_reasoning": state.get("route_reasoning", ""),
        "question_type": state.get("question_type", ""),
        "entities": state.get("entities", []),
        "attempts": state.get("attempt", 0),
        "risk": state.get("risk"),
        "graph_paths": state.get("graph_paths", []),
        "sources_reviewed": len(state.get("context_blocks", []) or []),
        "contains_identifiers": state.get("contains_identifiers", False),
        "degraded": state.get("degraded", False),
        "trace": state.get("trace", []),
        "llm_usage": USAGE.snapshot(),
        "provider": provider_info(),
    }
