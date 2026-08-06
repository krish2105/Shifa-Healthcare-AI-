"""Draft Composer and Final Composer nodes.

Split into two because they answer to different pressures. The draft is written to
be *checkable* — every claim carries a citation so the critic can decompose and
verify it. The final pass is written to be *readable*, and is forbidden from adding
content, so it cannot reintroduce anything the critic just cleared.

The draft is also permitted to abstain outright by returning INSUFFICIENT_EVIDENCE.
That gives the system a cheap escape before it spends a critic call on a draft that
its own author already knows is unsupported.
"""

from __future__ import annotations

import re
import time
from typing import Any

from app.agent.nodes.fusion import render_sources
from app.agent.state import AgentState, trace_event
from app.llm import router as llm
from app.llm.prompts import DRAFT_SYSTEM, FINAL_SYSTEM
from app.logging_conf import get_logger

log = get_logger("agent.composer")

INSUFFICIENT = "INSUFFICIENT_EVIDENCE"
_CITATION = re.compile(r"\[(\d{1,2})\]")


async def draft_node(state: AgentState) -> dict[str, Any]:
    t0 = time.perf_counter()
    blocks = state.get("context_blocks") or []
    query = state.get("normalized_query") or state["query"]
    attempt = state.get("attempt", 0)

    if not blocks:
        dt = (time.perf_counter() - t0) * 1000
        return {
            "draft": f"{INSUFFICIENT}: no guideline passages were retrieved for this question.",
            "attempt": attempt + 1,
            "trace": [trace_event("draft", "no_context", dt, attempt=attempt + 1)],
        }

    user_parts = [f"QUESTION: {query}", "", "SOURCES:", render_sources(blocks)]

    risk = state.get("risk")
    if risk:
        user_parts += [
            "",
            "PATIENT RISK CONTEXT (from the ED risk model — a statistical estimate, "
            "not a source you may cite as a guideline):",
            f"  risk score: {risk.get('risk_score')} ({risk.get('band')})",
            f"  model: {risk.get('model')}, top features: {risk.get('top_features')}",
        ]

    paths = state.get("graph_paths") or []
    if paths:
        user_parts += [
            "",
            "KNOWLEDGE-GRAPH RELATIONSHIPS (extracted from the sources above; "
            "verify against the source text before relying on any of them):",
            *[f"  - {p['describe']}" for p in paths[:5]],
        ]

    draft = await llm.complete(
        [
            {"role": "system", "content": DRAFT_SYSTEM},
            {"role": "user", "content": "\n".join(user_parts)},
        ],
        tier=llm.Tier.STRONG,
        max_tokens=900,
    )

    cited = sorted({int(m) for m in _CITATION.findall(draft)})
    dt = (time.perf_counter() - t0) * 1000
    log.info("draft.done", run_id=state["run_id"], attempt=attempt + 1, citations=len(cited))

    return {
        "draft": draft.strip(),
        "attempt": attempt + 1,
        "trace": [
            trace_event(
                "draft", "composed", dt,
                attempt=attempt + 1,
                chars=len(draft),
                citations_used=cited,
                sources_available=len(blocks),
                abstained=draft.strip().startswith(INSUFFICIENT),
            )
        ],
    }


async def final_node(state: AgentState) -> dict[str, Any]:
    """Polish a draft that passed the critic, and bind its citations."""
    t0 = time.perf_counter()
    draft = state.get("draft", "")
    blocks = state.get("context_blocks") or []

    final = await llm.complete(
        [
            {"role": "system", "content": FINAL_SYSTEM},
            {
                "role": "user",
                "content": f"DRAFT:\n{draft}\n\nSOURCES:\n{render_sources(blocks, max_chars=900)}",
            },
        ],
        tier=llm.Tier.STRONG,
        max_tokens=900,
    )
    final = final.strip() or draft

    # Guard against the polish pass dropping citations. If it did, keep the draft —
    # a slightly rougher answer that is traceable beats a smoother one that is not.
    draft_cites = {int(m) for m in _CITATION.findall(draft)}
    final_cites = {int(m) for m in _CITATION.findall(final)}
    citations_preserved = draft_cites.issubset(final_cites)
    if not citations_preserved and draft_cites:
        log.warning(
            "final.citations_dropped",
            run_id=state["run_id"], draft=sorted(draft_cites), final=sorted(final_cites),
        )
        final = draft
        final_cites = draft_cites

    # Only cited sources become citations. Listing uncited retrievals as sources
    # would overstate what the answer actually rests on.
    citations = [
        {
            "index": b["index"],
            "title": b["title"],
            "source": b["source"],
            "section": b["section"],
            "url": b["url"],
            "score": b["score"],
            "retriever": b["retriever"],
            "components": b.get("components", {}),
            "snippet": b["matched_text"][:320] + ("…" if len(b["matched_text"]) > 320 else ""),
        }
        for b in blocks
        if b["index"] in final_cites
    ]

    dt = (time.perf_counter() - t0) * 1000
    return {
        "answer": final,
        "citations": citations,
        "outcome": "answered",
        "trace": [
            trace_event(
                "final_composer", "composed", dt,
                citations_bound=len(citations),
                citations_preserved=citations_preserved,
                groundedness=state.get("groundedness"),
            )
        ],
    }
