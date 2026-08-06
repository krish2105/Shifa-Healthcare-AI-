"""Retriever nodes: dense+sparse vector search, graph traversal, and risk scoring.

These three run concurrently when the planner selects more than one. Each writes to
its own state key, so there is no merge conflict and no ordering dependency — the
fusion node is where their outputs actually meet.
"""

from __future__ import annotations

import time
from typing import Any

from app.agent.state import AgentState, trace_event
from app.config import settings
from app.logging_conf import get_logger
from app.retrieval.types import Chunk, SearchHit

log = get_logger("agent.retrievers")


# ------------------------------------------------------------------ vector


async def vector_retriever_node(state: AgentState) -> dict[str, Any]:
    """Hybrid dense + sparse retrieval.

    Both run on every invocation of this node. They fail differently — dense search
    misses rare drug names and codes it has smoothed away, BM25 misses paraphrase —
    and the fusion node reconciles them by rank. Running only one would be cheaper
    per query but would leave a known, systematic recall hole.
    """
    t0 = time.perf_counter()
    query = state.get("search_query") or state["query"]

    from app.retrieval.bm25 import get_bm25
    from app.retrieval.embedder import embed_query
    from app.retrieval.store import get_store

    store = get_store()
    k = settings.retrieval_candidate_k

    dense_hits: list[SearchHit] = []
    try:
        dense_hits = store.search(embed_query(query), k)
    except Exception as exc:  # noqa: BLE001
        log.error("retriever.dense_failed", error=str(exc)[:200])

    sparse_hits: list[SearchHit] = []
    bm = get_bm25()
    if bm is not None:
        try:
            sparse_hits = bm.search(query, k)
        except Exception as exc:  # noqa: BLE001
            log.error("retriever.sparse_failed", error=str(exc)[:200])

    dt = (time.perf_counter() - t0) * 1000
    return {
        "dense_hits": dense_hits,
        "sparse_hits": sparse_hits,
        "trace": [
            trace_event(
                "vector_retriever", "retrieved", dt,
                dense=len(dense_hits), sparse=len(sparse_hits),
                backend=store.backend,
                top_dense_score=round(dense_hits[0].score, 4) if dense_hits else None,
            )
        ],
    }


# ------------------------------------------------------------------- graph


async def graph_retriever_node(state: AgentState) -> dict[str, Any]:
    """Multi-hop traversal over the condition-drug-symptom graph.

    Seeds come from the entities intake extracted, falling back to label matching
    against the raw query. Traversal returns paths; each path carries the chunk ids
    of the passages its nodes were extracted from, and those chunks are what get
    handed to fusion — so a graph result is still a citable piece of text, never a
    bare assertion that some edge exists.
    """
    t0 = time.perf_counter()

    from app.retrieval.graph_store import get_graph

    kg = get_graph()
    if kg is None:
        dt = (time.perf_counter() - t0) * 1000
        return {
            "graph_hits": [],
            "graph_paths": [],
            "trace": [
                trace_event("graph_retriever", "skipped", dt,
                            reason="no knowledge graph built — run scripts/build_graph.py")
            ],
        }

    entities = state.get("entities") or []
    query = state.get("normalized_query") or state["query"]

    seeds: list[str] = []
    for ent in entities:
        seeds.extend(kg.match_entities(ent, limit=3))
    if not seeds:
        seeds = kg.match_entities(query, limit=6)
    seeds = list(dict.fromkeys(seeds))[:8]

    paths = kg.traverse(seeds) if seeds else []

    # Resolve path chunk ids back to text so graph evidence is citable.
    chunk_ids: list[str] = []
    for p in paths[: settings.retrieval_top_k]:
        chunk_ids.extend(p.chunk_ids)
    chunk_ids = list(dict.fromkeys(chunk_ids))[: settings.retrieval_candidate_k]

    graph_hits = _resolve_chunks(chunk_ids)

    path_payload = [
        {"describe": p.describe(), "nodes": p.nodes, "score": round(p.score, 4)}
        for p in paths[:8]
    ]

    dt = (time.perf_counter() - t0) * 1000
    log.info("retriever.graph", run_id=state["run_id"], seeds=len(seeds), paths=len(paths))
    return {
        "graph_hits": graph_hits,
        "graph_paths": path_payload,
        "trace": [
            trace_event(
                "graph_retriever", "traversed", dt,
                seeds=seeds[:6], paths=len(paths), chunks=len(graph_hits),
                hops=settings.graph_max_hops,
            )
        ],
    }


def _resolve_chunks(chunk_ids: list[str]) -> list[SearchHit]:
    """Look up chunk text by id from whichever store is active."""
    if not chunk_ids:
        return []

    from app.retrieval.bm25 import get_bm25
    from app.retrieval.store import LocalVectorStore, PgVectorStore, get_store

    store = get_store()
    lookup: dict[str, Chunk] = {}

    if isinstance(store, LocalVectorStore):
        lookup = {c.chunk_id: c for c in store._chunks if c.chunk_id in set(chunk_ids)}  # noqa: SLF001
    elif isinstance(store, PgVectorStore):
        try:
            with store.pool.connection() as conn, conn.cursor() as cur:
                cur.execute(
                    """SELECT chunk_id,parent_id,doc_id,title,source,section,url,token_count,text
                       FROM chunks WHERE chunk_id = ANY(%s)""",
                    (chunk_ids,),
                )
                for r in cur.fetchall():
                    lookup[r[0]] = Chunk(
                        chunk_id=r[0], parent_id=r[1], doc_id=r[2], title=r[3] or "",
                        source=r[4] or "", section=r[5] or "", url=r[6] or "",
                        token_count=r[7] or 0, text=r[8],
                    )
        except Exception as exc:  # noqa: BLE001
            log.error("retriever.graph_lookup_failed", error=str(exc)[:200])

    if not lookup:
        bm = get_bm25()
        if bm is not None:
            wanted = set(chunk_ids)
            lookup = {c.chunk_id: c for c in bm._chunks if c.chunk_id in wanted}  # noqa: SLF001

    return [
        SearchHit(chunk=lookup[cid], score=1.0 / (1 + i), retriever="graph", rank=i)
        for i, cid in enumerate(chunk_ids)
        if cid in lookup
    ]


# -------------------------------------------------------------------- risk


async def risk_node(state: AgentState) -> dict[str, Any]:
    """Attach the ED risk-stratification score for a referenced patient.

    Runs only on the patient-context route. Returns None rather than a default score
    when the model or the patient is unavailable — an invented risk number is worse
    than an absent one, and the composer is built to handle absence.
    """
    t0 = time.perf_counter()
    patient_id = state.get("patient_id")

    if not patient_id:
        dt = (time.perf_counter() - t0) * 1000
        return {
            "risk": None,
            "trace": [trace_event("risk_node", "skipped", dt, reason="no patient_id supplied")],
        }

    from app.risk.predict import predict_risk

    try:
        risk = predict_risk(patient_id)
    except Exception as exc:  # noqa: BLE001
        log.error("risk.failed", error=str(exc)[:200])
        risk = None

    dt = (time.perf_counter() - t0) * 1000
    return {
        "risk": risk,
        "trace": [
            trace_event(
                "risk_node", "scored" if risk else "unavailable", dt,
                patient_id=patient_id,
                score=risk.get("risk_score") if risk else None,
                model=risk.get("model") if risk else None,
            )
        ],
    }
