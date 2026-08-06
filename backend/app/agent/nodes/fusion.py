"""Context Fusion node — reconcile the retrievers into one citable evidence set.

Order matters here:

1. **RRF** merges dense, sparse and graph rankings by rank, not score. Their score
   scales are not comparable, so any weighted blend would need a normalization
   scheme and a weight we cannot honestly tune at this data scale.
2. **Dedup** collapses chunks sharing a parent. Two children of one passage are one
   piece of evidence; counting them twice inflates apparent support.
3. **MMR** trades a little relevance for coverage. Guideline corpora restate the
   same recommendation across many documents, and a top-k of five paraphrases gives
   the composer no more information than one while looking like strong corroboration.
4. **Parent expansion** swaps each retrieved child for its parent passage. We search
   narrow and compose wide.

The weights favour dense slightly over sparse, and graph lowest — graph edges are
LLM-extracted and noisier than either text index, so they inform the ranking without
dominating it.
"""

from __future__ import annotations

import time
from typing import Any

from app.agent.state import AgentState, trace_event
from app.config import settings
from app.logging_conf import get_logger
from app.retrieval.fusion import deduplicate, mmr_select, reciprocal_rank_fusion

log = get_logger("agent.fusion")

RETRIEVER_WEIGHTS = {"dense": 1.0, "sparse": 0.8, "graph": 0.6}


async def fusion_node(state: AgentState) -> dict[str, Any]:
    t0 = time.perf_counter()

    dense = state.get("dense_hits") or []
    sparse = state.get("sparse_hits") or []
    graph = state.get("graph_hits") or []

    lists = []
    weights = []
    for hits, key in ((dense, "dense"), (sparse, "sparse"), (graph, "graph")):
        if hits:
            lists.append(hits)
            weights.append(RETRIEVER_WEIGHTS[key])

    if not lists:
        dt = (time.perf_counter() - t0) * 1000
        return {
            "fused_hits": [],
            "context_blocks": [],
            "trace": [trace_event("fusion", "empty", dt, reason="no retriever returned results")],
        }

    fused = reciprocal_rank_fusion(lists, weights=weights)
    before_dedup = len(fused)
    fused = deduplicate(fused)

    selected = mmr_select(fused, None, k=settings.retrieval_top_k)

    context_blocks = _expand_to_parents(selected)

    dt = (time.perf_counter() - t0) * 1000
    log.info(
        "fusion.done",
        run_id=state["run_id"], fused=before_dedup,
        deduped=len(fused), selected=len(selected),
    )
    return {
        "fused_hits": selected,
        "context_blocks": context_blocks,
        "trace": [
            trace_event(
                "fusion", "fused", dt,
                inputs={"dense": len(dense), "sparse": len(sparse), "graph": len(graph)},
                fused=before_dedup,
                after_dedup=len(fused),
                selected=len(selected),
                mmr_lambda=settings.mmr_lambda,
                rrf_k=settings.rrf_k,
            )
        ],
    }


def _expand_to_parents(hits: list[Any]) -> list[dict[str, Any]]:
    """Swap each retrieved child chunk for its parent passage.

    Falls back to the child's own text when the parent is missing, so a partial
    index degrades to narrower context rather than to an empty block.
    """
    from app.retrieval.store import get_store

    store = get_store()
    blocks: list[dict[str, Any]] = []

    for i, hit in enumerate(hits, start=1):
        c = hit.chunk
        parent_text: str | None = None
        try:
            parent_text = store.get_parent(c.parent_id)
        except Exception:  # noqa: BLE001 — a missing parent must not fail the request
            parent_text = None

        blocks.append(
            {
                "index": i,
                "chunk_id": c.chunk_id,
                "parent_id": c.parent_id,
                "title": c.title or c.doc_id,
                "source": c.source,
                "section": c.section,
                "url": c.url,
                "retriever": hit.retriever,
                "components": hit.components,
                "score": round(hit.score, 4),
                "text": parent_text or c.text,
                "matched_text": c.text,
                "expanded": bool(parent_text and parent_text != c.text),
            }
        )
    return blocks


def render_sources(blocks: list[dict[str, Any]], max_chars: int = 1800) -> str:
    """Format context blocks as the numbered SOURCES section the prompts expect."""
    parts: list[str] = []
    for b in blocks:
        header = f"[{b['index']}] {b['title']}"
        if b.get("section"):
            header += f" — {b['section']}"
        if b.get("source"):
            header += f" ({b['source']})"
        body = b["text"][:max_chars]
        if len(b["text"]) > max_chars:
            body += " …"
        parts.append(f"{header}\n{body}")
    return "\n\n".join(parts)
