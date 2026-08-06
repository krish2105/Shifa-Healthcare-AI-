"""RAGAS-definition retrieval metrics, implemented natively.

**Why not import RAGAS.** RAGAS pulls a LangChain provider stack and expects an
OpenAI-shaped client; wiring it through our LiteLLM free-tier chain means fighting
its abstractions, and it makes the safety-critical critic path depend on a heavy
third-party evaluation library. The metric *definitions* are what matter, and they
are short. We implement those and keep `ragas` as an optional extra
(`pip install -e ".[ragas]"`) purely to cross-check our numbers.

Metrics (standard definitions):

* **faithfulness** — of the atomic claims in the answer, the fraction entailed by
  the retrieved context. This is the hallucination measure, and the one the critic
  gates on.
* **answer_relevance** — does the answer address the question asked.
* **context_precision** — of the retrieved chunks, the fraction actually useful.
* **context_recall** — of the ground-truth answer's claims, the fraction the
  retrieved context covers. Needs a reference answer, so it is eval-only.

Each has an LLM judge and a cheap lexical fallback. The fallback exists because the
critic must return *something* when the LLM chain is rate-limited, and a degraded
score that is visibly conservative is safer than an exception that drops the request
or a default of 1.0 that waves an ungrounded answer through.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from app.llm import router as llm
from app.llm.prompts import CRITIC_SYSTEM
from app.logging_conf import get_logger

log = get_logger("eval.metrics")

_SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")
_WORD = re.compile(r"[a-z0-9]+(?:[-/][a-z0-9]+)*")
_CITATION = re.compile(r"\[\d{1,2}\]")

_STOP = frozenset(
    ["a", "an", "the", "and", "or", "of", "to", "in", "for", "on", "at", "by", "with", "from", "as", "is", "are", "was", "were", "be", "been", "this", "that", "these", "those", "it", "its", "which", "what", "when", "where", "who", "how", "not", "no", "should", "may", "can", "if", "then", "than", "there", "their", "them", "they", "we", "you", "i", "he", "she", "his", "her"]
)


def _tokens(text: str) -> set[str]:
    return {w for w in _WORD.findall(text.lower()) if w not in _STOP and len(w) > 2}


def split_claims(answer: str) -> list[str]:
    """Split an answer into atomic claims, dropping citation markers and boilerplate."""
    clean = _CITATION.sub("", answer)
    clean = re.sub(r"^\s*[-*•]\s*", "", clean, flags=re.M)
    sentences = [s.strip() for s in _SENT.split(clean) if s.strip()]
    return [s for s in sentences if len(s.split()) >= 4]


@dataclass
class FaithfulnessResult:
    score: float
    total_claims: int
    supported_claims: int
    unsupported: list[dict[str, str]] = field(default_factory=list)
    reasoning: str = ""
    method: str = "llm"

    def as_dict(self) -> dict[str, Any]:
        return {
            "faithfulness": round(self.score, 4),
            "total_claims": self.total_claims,
            "supported_claims": self.supported_claims,
            "unsupported": self.unsupported[:6],
            "reasoning": self.reasoning,
            "method": self.method,
        }


def lexical_faithfulness(answer: str, contexts: list[str]) -> FaithfulnessResult:
    """Token-overlap entailment proxy.

    Crude by construction: it measures whether a claim's content words appear in the
    context, which catches wholesale fabrication but not a subtly altered dose or an
    inverted negation. It is a floor, not a verdict, and it is labelled `method
    ="lexical"` wherever it is reported so no one mistakes it for the real check.
    """
    claims = split_claims(answer)
    if not claims:
        return FaithfulnessResult(0.0, 0, 0, reasoning="no verifiable claims found", method="lexical")

    ctx_tokens = set()
    for c in contexts:
        ctx_tokens |= _tokens(c)

    supported = 0
    unsupported: list[dict[str, str]] = []
    for claim in claims:
        ct = _tokens(claim)
        if not ct:
            continue
        coverage = len(ct & ctx_tokens) / len(ct)
        if coverage >= 0.6:
            supported += 1
        else:
            unsupported.append(
                {"claim": claim[:180], "why": f"only {coverage:.0%} of content words appear in context"}
            )

    return FaithfulnessResult(
        score=supported / len(claims),
        total_claims=len(claims),
        supported_claims=supported,
        unsupported=unsupported,
        reasoning="lexical overlap fallback — no LLM judge available",
        method="lexical",
    )


async def faithfulness(answer: str, contexts: list[str]) -> FaithfulnessResult:
    """LLM-judged faithfulness, falling back to lexical on provider failure."""
    if not answer.strip() or not contexts:
        return FaithfulnessResult(0.0, 0, 0, reasoning="empty answer or context", method="none")

    sources = "\n\n".join(f"[{i}] {c[:1500]}" for i, c in enumerate(contexts, start=1))

    try:
        result = await llm.complete_json(
            [
                {"role": "system", "content": CRITIC_SYSTEM},
                {"role": "user", "content": f"SOURCES:\n{sources}\n\nANSWER:\n{answer}"},
            ],
            tier=llm.Tier.STRONG,
            default={},
            max_tokens=700,
        )
    except Exception as exc:  # noqa: BLE001
        log.warning("metrics.faithfulness_llm_failed", error=str(exc)[:180])
        return lexical_faithfulness(answer, contexts)

    if not result:
        return lexical_faithfulness(answer, contexts)

    try:
        total = int(result.get("total_claims") or 0)
        supported = int(result.get("supported_claims") or 0)
        raw_score = result.get("faithfulness")
        # Prefer the ratio the judge's own counts imply over the score it reported.
        # Models routinely return a plausible-looking score that contradicts their
        # claim tally; the counts are the more reliable signal.
        score = (supported / total) if total > 0 else (float(raw_score) if raw_score is not None else 0.0)
        score = max(0.0, min(1.0, score))
    except (TypeError, ValueError, ZeroDivisionError):
        return lexical_faithfulness(answer, contexts)

    unsupported = []
    for u in result.get("unsupported") or []:
        if isinstance(u, dict):
            unsupported.append({"claim": str(u.get("claim", ""))[:180], "why": str(u.get("why", ""))[:180]})
        elif isinstance(u, str):
            unsupported.append({"claim": u[:180], "why": ""})

    return FaithfulnessResult(
        score=score,
        total_claims=total,
        supported_claims=supported,
        unsupported=unsupported,
        reasoning=str(result.get("reasoning") or "")[:300],
        method="llm",
    )


async def answer_relevance(question: str, answer: str) -> float:
    """How well the answer addresses the question actually asked."""
    if not answer.strip():
        return 0.0

    result = await llm.complete_json(
        [
            {
                "role": "system",
                "content": (
                    "Rate how directly the ANSWER addresses the QUESTION. Return JSON: "
                    '{"relevance": <0.0-1.0>, "reasoning": "<one sentence>"}. '
                    "Judge only relevance to the question, not factual accuracy and not "
                    "whether the answer is well-sourced."
                ),
            },
            {"role": "user", "content": f"QUESTION: {question}\n\nANSWER: {answer[:2500]}"},
        ],
        tier=llm.Tier.FAST,
        default={"relevance": 0.0},
        max_tokens=200,
    )
    try:
        return max(0.0, min(1.0, float(result.get("relevance", 0.0))))
    except (TypeError, ValueError):
        # Lexical fallback: content-word overlap between question and answer.
        q, a = _tokens(question), _tokens(answer)
        return len(q & a) / len(q) if q else 0.0


async def context_precision(question: str, contexts: list[str]) -> float:
    """Fraction of retrieved chunks that are actually useful for the question.

    Low precision with a good answer means we are paying to stuff the prompt with
    passages the composer ignored — a cost problem, not a correctness one. Reported
    because it is what tells you whether top_k is set sensibly.
    """
    if not contexts:
        return 0.0

    result = await llm.complete_json(
        [
            {
                "role": "system",
                "content": (
                    "For each numbered CONTEXT, decide whether it is useful for answering "
                    'the QUESTION. Return JSON: {"useful": [<1-based indices>]}. '
                    "Be strict: topically adjacent is not useful."
                ),
            },
            {
                "role": "user",
                "content": f"QUESTION: {question}\n\nCONTEXTS:\n"
                + "\n\n".join(f"[{i}] {c[:700]}" for i, c in enumerate(contexts, start=1)),
            },
        ],
        tier=llm.Tier.FAST,
        default={},
        max_tokens=250,
    )

    useful = result.get("useful")
    if not isinstance(useful, list):
        q = _tokens(question)
        if not q:
            return 0.0
        hits = sum(1 for c in contexts if len(q & _tokens(c)) / len(q) >= 0.3)
        return hits / len(contexts)

    valid = {i for i in useful if isinstance(i, int) and 1 <= i <= len(contexts)}
    return len(valid) / len(contexts)


async def context_recall(reference_answer: str, contexts: list[str]) -> float:
    """Fraction of the reference answer's claims covered by the retrieved context.

    Eval-only: it needs a ground-truth answer, which production queries do not have.
    This is the metric that separates "the generator was weak" from "retrieval never
    surfaced the evidence" — without it, a bad score is unattributable.
    """
    if not reference_answer.strip() or not contexts:
        return 0.0

    claims = split_claims(reference_answer)
    if not claims:
        claims = [reference_answer]

    result = await llm.complete_json(
        [
            {
                "role": "system",
                "content": (
                    "For each numbered CLAIM, decide whether the CONTEXT supports it. "
                    'Return JSON: {"supported": [<1-based claim indices>]}.'
                ),
            },
            {
                "role": "user",
                "content": "CONTEXT:\n"
                + "\n\n".join(c[:900] for c in contexts[:8])
                + "\n\nCLAIMS:\n"
                + "\n".join(f"[{i}] {c}" for i, c in enumerate(claims, start=1)),
            },
        ],
        tier=llm.Tier.FAST,
        default={},
        max_tokens=250,
    )

    supported = result.get("supported")
    if not isinstance(supported, list):
        ctx_tokens: set[str] = set()
        for c in contexts:
            ctx_tokens |= _tokens(c)
        hits = sum(1 for cl in claims if len(_tokens(cl) & ctx_tokens) / max(len(_tokens(cl)), 1) >= 0.6)
        return hits / len(claims)

    valid = {i for i in supported if isinstance(i, int) and 1 <= i <= len(claims)}
    return len(valid) / len(claims)
