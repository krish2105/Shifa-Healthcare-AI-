#!/usr/bin/env python
"""Retrieval diagnostics that need no LLM.

The README makes two load-bearing claims about retrieval design:

  1. dense and sparse retrieval fail differently, so fusing them is worth an
     extra index rather than being cargo-culted;
  2. guideline corpora are redundant enough that MMR earns its place.

Both are measurable without a language model, and this script measures them
against real clinical questions drawn from PubMedQA.

**What this is not.** These are *diagnostics*, not accuracy. Measuring retrieval
recall honestly requires gold relevance labels for THIS corpus — which questions
should retrieve which of our 53,026 chunks — and no such labels exist. Nothing
here is a substitute for the benchmark accuracy in RESULTS.md §8; inventing a
recall proxy and presenting it as retrieval quality would be exactly the kind of
number this project refuses to publish elsewhere.

Usage:
    python scripts/eval_retrieval.py --n 100
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import settings
from app.logging_conf import get_logger
from app.retrieval.bm25 import get_bm25, tokenize
from app.retrieval.embedder import embed_query
from app.retrieval.fusion import deduplicate, mmr_select, reciprocal_rank_fusion
from app.retrieval.store import get_store
from app.retrieval.types import SearchHit

log = get_logger("eval.retrieval")

SEED = 42


def load_queries(n: int) -> list[str]:
    """Real clinical questions. PubMedQA questions are used purely as query text —
    its own contexts are irrelevant here since we retrieve against our corpus."""
    from datasets import load_dataset

    ds = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train")
    rng = random.Random(SEED)
    idx = rng.sample(range(len(ds)), min(n, len(ds)))
    return [ds[i]["question"] for i in idx]


def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    union = len(a | b)
    return len(a & b) / union if union else 0.0


def mean_pairwise_redundancy(hits: list[SearchHit]) -> float:
    """Mean pairwise token-Jaccard across a result set.

    A high value means the set says the same thing several times over — which is
    the failure mode MMR exists to correct.
    """
    if len(hits) < 2:
        return 0.0
    toks = [set(tokenize(h.chunk.text)) for h in hits]
    sims = [
        jaccard(toks[i], toks[j])
        for i in range(len(toks))
        for j in range(i + 1, len(toks))
    ]
    return statistics.mean(sims) if sims else 0.0


def pct(xs: list[float], p: float) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    k = min(len(s) - 1, round((len(s) - 1) * p))
    return s[k]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100, help="Queries to evaluate.")
    ap.add_argument("--top-k", type=int, default=settings.retrieval_top_k)
    args = ap.parse_args()

    settings.ensure_dirs()

    store = get_store()
    bm = get_bm25()

    if store.count() == 0:
        log.error("eval.empty_index", hint="Run scripts/embed_index.py first.")
        return 1
    if bm is None:
        log.error("eval.no_bm25", hint="Run scripts/embed_index.py (without --skip-bm25).")
        return 1

    log.info("eval.start", queries=args.n, chunks=store.count(), bm25_chunks=bm.count())
    queries = load_queries(args.n)

    overlap_at10: list[float] = []
    dense_only: list[float] = []
    sparse_only: list[float] = []
    both: list[float] = []
    red_before: list[float] = []
    red_after: list[float] = []
    lat_dense: list[float] = []
    lat_sparse: list[float] = []
    lat_fuse: list[float] = []
    sparse_empty = 0

    k_cand = settings.retrieval_candidate_k

    for i, q in enumerate(queries):
        t0 = time.perf_counter()
        dense = store.search(embed_query(q), k_cand)
        lat_dense.append((time.perf_counter() - t0) * 1000)

        t0 = time.perf_counter()
        sparse = bm.search(q, k_cand)
        lat_sparse.append((time.perf_counter() - t0) * 1000)
        if not sparse:
            sparse_empty += 1

        d10 = {h.chunk.chunk_id for h in dense[:10]}
        s10 = {h.chunk.chunk_id for h in sparse[:10]}
        overlap_at10.append(jaccard(d10, s10))

        t0 = time.perf_counter()
        lists, weights = [], []
        if dense:
            lists.append(dense)
            weights.append(1.0)
        if sparse:
            lists.append(sparse)
            weights.append(0.8)
        fused = deduplicate(reciprocal_rank_fusion(lists, weights=weights))
        pre_mmr = fused[: args.top_k]
        selected = mmr_select(fused, None, k=args.top_k)
        lat_fuse.append((time.perf_counter() - t0) * 1000)

        # Provenance of the final set: what did each retriever contribute?
        dall = {h.chunk.chunk_id for h in dense}
        sall = {h.chunk.chunk_id for h in sparse}
        final = [h.chunk.chunk_id for h in selected]
        if final:
            dense_only.append(sum(1 for c in final if c in dall and c not in sall) / len(final))
            sparse_only.append(sum(1 for c in final if c in sall and c not in dall) / len(final))
            both.append(sum(1 for c in final if c in dall and c in sall) / len(final))

        red_before.append(mean_pairwise_redundancy(pre_mmr))
        red_after.append(mean_pairwise_redundancy(selected))

        if (i + 1) % 25 == 0:
            log.info("eval.progress", done=i + 1, of=len(queries))

    def summarize(name: str, xs: list[float]) -> dict:
        return {
            "mean": round(statistics.mean(xs), 4) if xs else 0.0,
            "median": round(statistics.median(xs), 4) if xs else 0.0,
            "p95": round(pct(xs, 0.95), 4),
        }

    redundancy_drop = (
        round((statistics.mean(red_before) - statistics.mean(red_after)) / statistics.mean(red_before), 4)
        if red_before and statistics.mean(red_before) > 0
        else 0.0
    )

    report = {
        "queries": len(queries),
        "query_source": "PubMedQA questions (query text only)",
        "index": {
            "chunks": store.count(),
            "backend": store.backend,
            "bm25_chunks": bm.count(),
            "top_k": args.top_k,
            "candidate_k": k_cand,
        },
        "dense_vs_sparse_complementarity": {
            "jaccard_overlap_at_10": summarize("overlap", overlap_at10),
            "note": (
                "Jaccard of the dense top-10 and BM25 top-10 chunk sets. A low value "
                "means the two retrievers surface largely different evidence, which is "
                "the premise of fusing them. A value near 1.0 would mean the sparse "
                "index is redundant and could be dropped."
            ),
        },
        "final_set_provenance": {
            "from_dense_only": summarize("dense_only", dense_only),
            "from_sparse_only": summarize("sparse_only", sparse_only),
            "from_both": summarize("both", both),
            "sparse_returned_nothing": sparse_empty,
            "note": (
                "Share of the final top-k contributed uniquely by each retriever. "
                "Chunks from sparse_only would never have been seen by a dense-only "
                "system."
            ),
        },
        "mmr_redundancy_reduction": {
            "mean_pairwise_similarity_before": round(statistics.mean(red_before), 4) if red_before else 0.0,
            "mean_pairwise_similarity_after": round(statistics.mean(red_after), 4) if red_after else 0.0,
            "relative_reduction": redundancy_drop,
            "lambda": settings.mmr_lambda,
            "note": (
                "Mean pairwise token-Jaccard within the returned set, before and after "
                "MMR. A drop means MMR replaced near-duplicate passages with distinct "
                "ones. No drop would mean MMR is pure latency."
            ),
        },
        "latency_ms": {
            "dense": summarize("dense", lat_dense),
            "sparse": summarize("sparse", lat_sparse),
            "fusion_and_mmr": summarize("fuse", lat_fuse),
        },
        "caveat": (
            "Diagnostics, not accuracy. Honest recall would need gold relevance labels "
            "for this corpus, which do not exist. See RESULTS.md section 8 for answer "
            "accuracy, which requires an LLM."
        ),
    }

    out = settings.artifacts_dir / "retrieval_diagnostics.json"
    out.write_text(json.dumps(report, indent=2))
    log.info("eval.complete", path=str(out))
    print("\n" + json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
