"""Result fusion: Reciprocal Rank Fusion, then MMR for diversity.

**Why RRF and not weighted score blending.** BM25 scores are unbounded and corpus-
dependent; cosine scores sit in [-1, 1]. Blending them requires a normalization
scheme plus a weight, and both need tuning against a labelled set we do not have at
this scale. RRF discards magnitudes and uses rank only:

    score(d) = sum over retrievers of 1 / (k + rank(d))

It has one parameter, k, which mostly controls how sharply top ranks dominate, and
it is robust to the score-scale mismatch by construction. Choosing it is a decision
about what we can honestly tune, not just a default.

**Why MMR after fusion.** Guideline corpora are highly redundant — the same
recommendation is restated across many documents. Rank-ordered top-k tends to return
five phrasings of one idea, which inflates apparent support while giving the composer
no new information. MMR trades a little relevance for coverage, so the critic sees
genuinely distinct evidence.
"""

from __future__ import annotations

import numpy as np

from app.config import settings
from app.retrieval.types import SearchHit


def reciprocal_rank_fusion(
    result_lists: list[list[SearchHit]],
    *,
    k: int | None = None,
    weights: list[float] | None = None,
) -> list[SearchHit]:
    rrf_k = k if k is not None else settings.rrf_k
    if not result_lists:
        return []
    w = weights or [1.0] * len(result_lists)
    if len(w) != len(result_lists):
        raise ValueError("weights length must match result_lists length")

    accum: dict[str, float] = {}
    best: dict[str, SearchHit] = {}
    parts: dict[str, dict[str, float]] = {}

    for weight, hits in zip(w, result_lists, strict=True):
        for rank, hit in enumerate(hits):
            cid = hit.chunk.chunk_id
            contribution = weight / (rrf_k + rank + 1)
            accum[cid] = accum.get(cid, 0.0) + contribution
            parts.setdefault(cid, {})[hit.retriever] = round(hit.score, 4)
            # Keep the highest-ranked instance so snippets come from the strongest match.
            if cid not in best or rank < best[cid].rank:
                best[cid] = hit

    ordered = sorted(accum.items(), key=lambda kv: -kv[1])
    fused: list[SearchHit] = []
    for rank, (cid, score) in enumerate(ordered):
        h = best[cid]
        fused.append(
            SearchHit(
                chunk=h.chunk,
                score=float(score),
                retriever="fused",
                rank=rank,
                components=parts.get(cid, {}),
            )
        )
    return fused


def mmr_select(
    hits: list[SearchHit],
    vectors: np.ndarray | None,
    *,
    k: int,
    lambda_mult: float | None = None,
) -> list[SearchHit]:
    """Maximal Marginal Relevance over already-ranked hits.

    `vectors[i]` must be the (normalized) embedding of `hits[i]`. When vectors are
    unavailable we fall back to token-overlap similarity, which is cruder but still
    removes near-duplicate passages — the dominant redundancy in guideline text.
    """
    lam = settings.mmr_lambda if lambda_mult is None else lambda_mult
    if not hits:
        return []
    k = min(k, len(hits))
    if k == len(hits):
        return hits

    if vectors is not None and len(vectors) == len(hits):
        sim_matrix = vectors @ vectors.T
    else:
        sim_matrix = _token_similarity_matrix([h.chunk.text for h in hits])

    # Relevance proxy: position in the fused ranking, decayed. Using rank rather than
    # raw score keeps MMR on the same footing as RRF — magnitudes stay out of it.
    relevance = np.array([1.0 / (1 + h.rank) for h in hits], dtype=np.float32)

    selected: list[int] = [0]
    candidates = set(range(1, len(hits)))

    while len(selected) < k and candidates:
        best_idx, best_val = -1, -np.inf
        for i in candidates:
            redundancy = max(float(sim_matrix[i][j]) for j in selected)
            val = lam * float(relevance[i]) - (1 - lam) * redundancy
            if val > best_val:
                best_val, best_idx = val, i
        if best_idx < 0:
            break
        selected.append(best_idx)
        candidates.discard(best_idx)

    out: list[SearchHit] = []
    for new_rank, i in enumerate(selected):
        h = hits[i]
        out.append(
            SearchHit(
                chunk=h.chunk, score=h.score, retriever=h.retriever,
                rank=new_rank, components=h.components,
            )
        )
    return out


def _token_similarity_matrix(texts: list[str]) -> np.ndarray:
    from app.retrieval.bm25 import tokenize

    sets = [set(tokenize(t)) for t in texts]
    n = len(sets)
    m = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(i, n):
            a, b = sets[i], sets[j]
            union = len(a | b)
            jac = len(a & b) / union if union else 0.0
            m[i][j] = m[j][i] = jac
    return m


def deduplicate(hits: list[SearchHit]) -> list[SearchHit]:
    """Drop exact chunk repeats and collapse hits sharing a parent passage.

    Two child chunks from the same parent are, for the composer's purposes, one piece
    of evidence — counting them twice would inflate the apparent breadth of support.
    """
    seen_chunks: set[str] = set()
    seen_parents: set[str] = set()
    out: list[SearchHit] = []
    for h in hits:
        cid, pid = h.chunk.chunk_id, h.chunk.parent_id
        if cid in seen_chunks:
            continue
        if pid and pid in seen_parents:
            continue
        seen_chunks.add(cid)
        if pid:
            seen_parents.add(pid)
        out.append(h)
    return out
