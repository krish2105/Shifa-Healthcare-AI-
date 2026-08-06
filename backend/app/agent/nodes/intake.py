"""Intake node — normalize the query and screen it for identifiers.

Two jobs, and the second one is the reason this node exists rather than being folded
into the planner. Shifa42 promises no PHI touches the system. That promise is only
credible if something actually checks, at the boundary, before the text reaches a
retriever, a log line, or a third-party LLM API. This node is that boundary.

The check is a regex screen plus an LLM opinion. Neither is sufficient alone — regex
misses free-text names, and the LLM is not deterministic — so a hit from either
raises the flag. A flagged query is still answered (this is a demo over open data,
not a live clinical system), but the flag is recorded in the audit trail and
surfaced in the API response, so the behaviour is observable rather than assumed.
"""

from __future__ import annotations

import re
import time
from typing import Any

from app.agent.state import AgentState, trace_event
from app.llm import router
from app.llm.prompts import INTAKE_SYSTEM
from app.logging_conf import get_logger

log = get_logger("agent.intake")

# Deliberately conservative. These match shapes that are unambiguously identifiers;
# broader patterns would fire on ordinary clinical text (dates, dosages, ages) and
# train the reader to ignore the flag.
_IDENTIFIER_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("ssn", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
    ("mrn", re.compile(r"\b(?:mrn|medical record (?:number|no))\b[:\s#]*([A-Z0-9]{5,})", re.I)),
    ("phone", re.compile(r"\b(?:\+?\d{1,2}[\s.-])?\(?\d{3}\)?[\s.-]\d{3}[\s.-]\d{4}\b")),
    ("email", re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]{2,}\b")),
    ("dob", re.compile(r"\b(?:dob|date of birth)\b[:\s]*\d{1,4}[-/]\d{1,2}[-/]\d{1,4}", re.I)),
    ("nhs", re.compile(r"\b\d{3}\s?\d{3}\s?\d{4}\b")),
]


def screen_identifiers(text: str) -> list[str]:
    """Return the names of identifier patterns present. Empty list means clean."""
    return [name for name, pattern in _IDENTIFIER_PATTERNS if pattern.search(text)]


async def intake_node(state: AgentState) -> dict[str, Any]:
    t0 = time.perf_counter()
    query = state["query"]

    regex_hits = screen_identifiers(query)

    result = await router.complete_json(
        [
            {"role": "system", "content": INTAKE_SYSTEM},
            {"role": "user", "content": query},
        ],
        tier=router.Tier.FAST,
        default={
            "normalized_query": query,
            "entities": [],
            "question_type": "general",
            "contains_identifiers": False,
        },
    )

    normalized = str(result.get("normalized_query") or query).strip() or query
    entities = [str(e) for e in (result.get("entities") or []) if str(e).strip()][:12]
    question_type = str(result.get("question_type") or "general")
    llm_flag = bool(result.get("contains_identifiers"))

    contains_identifiers = bool(regex_hits) or llm_flag
    if contains_identifiers:
        log.warning(
            "intake.identifiers_detected",
            run_id=state["run_id"], regex_hits=regex_hits, llm_flag=llm_flag,
        )

    dt = (time.perf_counter() - t0) * 1000
    return {
        "normalized_query": normalized,
        "search_query": normalized,
        "entities": entities,
        "question_type": question_type,
        "contains_identifiers": contains_identifiers,
        "degraded": router.IS_DEGRADED,
        "trace": [
            trace_event(
                "intake", "normalized", dt,
                entities=entities,
                question_type=question_type,
                identifiers_flagged=contains_identifiers,
                identifier_patterns=regex_hits,
            )
        ],
    }
