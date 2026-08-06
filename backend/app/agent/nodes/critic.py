"""Critic / Groundedness node, plus reformulation and escalation.

This is the safety-critical node. Everything upstream is about assembling a good
answer; this is the only place that asks whether the answer is *entitled* to be
shown, and it is the sole gate between the composer and the user.

Control flow it governs:

    critic → pass          → final composer
    critic → fail, retries left → reformulate → retrieve again
    critic → fail, exhausted    → escalate

Three properties are deliberate:

* **Failure escalates, it does not hedge.** There is no path that ships a
  low-confidence answer with a disclaimer attached. Hedged clinical answers get read
  as answers.
* **Errors fail closed.** If the critic itself throws, the score is 0.0, which
  routes to escalation. A verification step that fails open is not a verification
  step.
* **The best score across attempts is retained.** Reformulation can retrieve worse
  context than the first attempt, and the escalation message reports the best
  groundedness actually achieved rather than whichever attempt happened to be last.
"""

from __future__ import annotations

import time
from typing import Any, Literal

from app.agent.nodes.composer import INSUFFICIENT
from app.agent.state import AgentState, trace_event
from app.config import settings
from app.eval.metrics import faithfulness
from app.llm import router as llm
from app.llm.prompts import ESCALATION_TEMPLATE, REFORMULATE_SYSTEM
from app.logging_conf import get_logger

log = get_logger("agent.critic")


async def critic_node(state: AgentState) -> dict[str, Any]:
    t0 = time.perf_counter()
    draft = state.get("draft", "")
    blocks = state.get("context_blocks") or []
    prev_best = state.get("best_groundedness", 0.0)

    # The drafter already declared it cannot answer. Spending a strong-model critic
    # call to confirm that is pure cost.
    if draft.strip().startswith(INSUFFICIENT):
        reason = draft.split(":", 1)[1].strip() if ":" in draft else "the drafter found the sources insufficient"
        dt = (time.perf_counter() - t0) * 1000
        return {
            "groundedness": 0.0,
            "best_groundedness": prev_best,
            "critic_report": {"faithfulness": 0.0, "method": "drafter_abstained", "reasoning": reason},
            "escalation_reason": reason,
            "trace": [trace_event("critic", "drafter_abstained", dt, reason=reason)],
        }

    contexts = [b["text"] for b in blocks]

    try:
        result = await faithfulness(draft, contexts)
    except Exception as exc:  # noqa: BLE001
        # Fail closed. An unverifiable answer is treated exactly like an ungrounded one.
        log.error("critic.failed_closed", run_id=state["run_id"], error=str(exc)[:200])
        dt = (time.perf_counter() - t0) * 1000
        return {
            "groundedness": 0.0,
            "best_groundedness": prev_best,
            "critic_report": {"faithfulness": 0.0, "method": "error", "reasoning": str(exc)[:200]},
            "escalation_reason": "the groundedness check could not be completed",
            "trace": [trace_event("critic", "error_failed_closed", dt, error=str(exc)[:160])],
        }

    score = result.score
    passed = score >= settings.groundedness_threshold

    dt = (time.perf_counter() - t0) * 1000
    log.info(
        "critic.scored",
        run_id=state["run_id"], score=round(score, 4),
        threshold=settings.groundedness_threshold, passed=passed,
        attempt=state.get("attempt"), method=result.method,
    )

    return {
        "groundedness": score,
        "best_groundedness": max(prev_best, score),
        "critic_report": result.as_dict(),
        "escalation_reason": (
            "; ".join(u["claim"] for u in result.unsupported[:3]) if result.unsupported else ""
        ),
        "trace": [
            trace_event(
                "critic", "passed" if passed else "failed", dt,
                faithfulness=round(score, 4),
                threshold=settings.groundedness_threshold,
                supported_claims=result.total_claims and result.supported_claims,
                total_claims=result.total_claims,
                method=result.method,
                unsupported_count=len(result.unsupported),
                attempt=state.get("attempt"),
            )
        ],
    }


def critic_decision(state: AgentState) -> Literal["final_composer", "reformulate", "escalate"]:
    """Conditional edge out of the critic."""
    score = state.get("groundedness", 0.0)
    attempt = state.get("attempt", 0)

    if score >= settings.groundedness_threshold:
        return "final_composer"
    if attempt <= settings.max_reformulations:
        return "reformulate"
    return "escalate"


async def reformulate_node(state: AgentState) -> dict[str, Any]:
    """Rewrite the search query after a groundedness failure.

    Rewrites the *query*, not the answer. The failure mode being addressed is
    retrieval finding the wrong evidence, so re-drafting against the same context
    would just produce a differently-worded ungrounded answer.
    """
    t0 = time.perf_counter()
    original = state.get("normalized_query") or state["query"]
    report = state.get("critic_report") or {}
    unsupported = report.get("unsupported") or []
    missing = "; ".join(u.get("claim", "") for u in unsupported[:3]) or "insufficient supporting detail"

    blocks = state.get("context_blocks") or []
    context_summary = "; ".join(f"{b['title']} — {b.get('section') or 'n/a'}" for b in blocks[:5])

    result = await llm.complete_json(
        [
            {"role": "system", "content": REFORMULATE_SYSTEM},
            {
                "role": "user",
                "content": (
                    f"ORIGINAL QUESTION: {original}\n\n"
                    f"RETRIEVED CONTEXT (insufficient): {context_summary}\n\n"
                    f"WHAT WAS MISSING: {missing}"
                ),
            },
        ],
        tier=llm.Tier.FAST,
        default={"reformulated_query": original, "strategy": "fallback: reused original query"},
        max_tokens=300,
    )

    new_query = str(result.get("reformulated_query") or original).strip() or original
    strategy = str(result.get("strategy") or "")[:200]

    dt = (time.perf_counter() - t0) * 1000
    log.info("reformulate.done", run_id=state["run_id"], attempt=state.get("attempt"))

    return {
        "search_query": new_query,
        "reformulation_strategy": strategy,
        # Reset per-attempt retrieval state so the next pass starts clean rather than
        # accumulating hits from the attempt that already failed.
        "dense_hits": [],
        "sparse_hits": [],
        "graph_hits": [],
        "trace": [
            trace_event(
                "reformulate", "rewritten", dt,
                attempt=state.get("attempt"),
                new_query=new_query[:200],
                strategy=strategy,
                missing=missing[:200],
            )
        ],
    }


async def escalate_node(state: AgentState) -> dict[str, Any]:
    """Terminal state: return no answer and hand off to a human.

    Composed from a template rather than generated. A model asked to write a refusal
    tends to smuggle a partial answer into it, which defeats the entire point.
    """
    t0 = time.perf_counter()

    attempts = state.get("attempt", 0)
    blocks = state.get("context_blocks") or []
    best = state.get("best_groundedness", state.get("groundedness", 0.0))
    missing = state.get("escalation_reason") or "the retrieved guidelines did not address this question"

    report = state.get("critic_report") or {}
    reason_clause = ""
    if report.get("method") == "error":
        reason_clause = " because the verification step could not be completed"
    elif report.get("method") == "drafter_abstained":
        reason_clause = ""

    message = ESCALATION_TEMPLATE.format(
        reason_clause=reason_clause,
        attempts=attempts,
        attempt_word="attempt" if attempts == 1 else "attempts",
        n_sources=len(blocks),
        source_word="source" if len(blocks) == 1 else "sources",
        score=best,
        threshold=settings.groundedness_threshold,
        missing=missing,
    )

    dt = (time.perf_counter() - t0) * 1000
    log.warning(
        "escalate.triggered",
        run_id=state["run_id"], attempts=attempts,
        best_groundedness=round(best, 4), sources=len(blocks),
    )

    return {
        "answer": message,
        # Escalations cite nothing on purpose. Attaching sources to a refusal invites
        # the reader to treat the refusal as a partial answer with references.
        "citations": [],
        "outcome": "escalated",
        "escalation_reason": missing,
        "trace": [
            trace_event(
                "escalate", "escalated", dt,
                attempts=attempts,
                best_groundedness=round(best, 4),
                threshold=settings.groundedness_threshold,
                sources_reviewed=len(blocks),
                reason=missing[:200],
            )
        ],
    }
