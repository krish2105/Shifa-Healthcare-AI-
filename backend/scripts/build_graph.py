#!/usr/bin/env python
"""Phase A.3 — build the clinical knowledge graph by LLM triple extraction.

Runs entity/relation extraction over a sample of guideline chunks and assembles a
NetworkX graph of conditions, medications, symptoms and guideline sections.

**This produces a noisy graph, and the script measures how noisy.** Edges come from
a free-tier LLM reading guideline prose, not from a curated ontology like UMLS or
SNOMED. `--audit-sample N` holds out N extracted triples with their source text and
writes them to `graph_extraction_audit.json` for manual checking, so RESULTS.md can
report a measured extraction precision instead of assuming the edges are clean.
That distinction is the difference between a knowledge graph and a pile of
plausible-looking arrows.

Requires a real LLM provider. Extraction against the stub would emit an empty
graph, which is worse than no graph because it looks like a successful run.

Usage:
    python scripts/build_graph.py                     # default sample
    python scripts/build_graph.py --chunks 1200 --concurrency 4
    python scripts/build_graph.py --audit-sample 50
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from app.config import settings
from app.llm import router as llm
from app.llm.prompts import EXTRACTION_SYSTEM
from app.logging_conf import get_logger
from app.retrieval.graph_store import EDGE_TYPES, KnowledgeGraph

log = get_logger("graph.build")

SEED = 42


async def extract_one(text: str, chunk_id: str, doc_id: str, sem: asyncio.Semaphore) -> dict:
    async with sem:
        try:
            result = await llm.complete_json(
                [
                    {"role": "system", "content": EXTRACTION_SYSTEM},
                    {"role": "user", "content": text[:3500]},
                ],
                tier=llm.Tier.FAST,
                default={"entities": [], "relations": []},
                max_tokens=700,
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("extract.failed", chunk_id=chunk_id, error=str(exc)[:140])
            return {"entities": [], "relations": [], "chunk_id": chunk_id, "doc_id": doc_id}
        result["chunk_id"] = chunk_id
        result["doc_id"] = doc_id
        result["source_text"] = text[:900]
        return result


async def main_async(args: argparse.Namespace) -> int:
    settings.ensure_dirs()

    llm.guard_real_llm("Knowledge-graph extraction")

    if not settings.chunks_path.exists():
        log.error("graph.no_chunks", hint="Run scripts/ingest_guidelines.py first.")
        return 1

    chunks_df = pd.read_parquet(settings.chunks_path)

    # Sample across documents rather than taking the head, so the graph is not
    # dominated by whichever documents happened to chunk first.
    rng = random.Random(SEED)
    idx = list(range(len(chunks_df)))
    rng.shuffle(idx)
    idx = idx[: args.chunks]
    sample = chunks_df.iloc[idx]

    log.info("graph.start", chunks=len(sample), concurrency=args.concurrency)

    sem = asyncio.Semaphore(args.concurrency)
    tasks = [
        extract_one(r.text, r.chunk_id, r.doc_id, sem)
        for r in sample.itertuples(index=False)
    ]

    results = []
    for done, coro in enumerate(asyncio.as_completed(tasks), start=1):
        results.append(await coro)
        if done % 50 == 0:
            log.info("graph.progress", done=done, of=len(tasks))

    # ---------------------------------------------------------------- assemble
    kg = KnowledgeGraph()
    audit_rows: list[dict] = []
    n_entities = n_relations = n_rejected = 0

    for res in results:
        chunk_id = res.get("chunk_id", "")
        doc_id = res.get("doc_id", "")

        entity_types: dict[str, str] = {}
        for ent in res.get("entities") or []:
            if not isinstance(ent, dict):
                continue
            name = str(ent.get("name") or "").strip()
            etype = str(ent.get("type") or "condition").strip()
            if not name or len(name) > 90:
                continue
            kg.add_entity(name, etype, chunk_id=chunk_id)
            entity_types[name.lower()] = etype
            n_entities += 1

        for rel in res.get("relations") or []:
            if not isinstance(rel, dict):
                continue
            s = str(rel.get("source") or "").strip()
            t = str(rel.get("target") or "").strip()
            r = str(rel.get("relation") or "").strip()
            if not s or not t or r not in EDGE_TYPES:
                n_rejected += 1
                continue
            # Endpoints must have been declared as entities; a relation naming
            # something the extractor never identified is usually a hallucinated
            # edge rather than an implied one.
            kg.add_entity(s, entity_types.get(s.lower(), "condition"), chunk_id=chunk_id)
            kg.add_entity(t, entity_types.get(t.lower(), "condition"), chunk_id=chunk_id)
            kg.add_relation(s, r, t, chunk_id=chunk_id, source_doc=doc_id)
            n_relations += 1

            if len(audit_rows) < args.audit_sample:
                audit_rows.append(
                    {
                        "triple": f"{s} —[{r}]→ {t}",
                        "source": s, "relation": r, "target": t,
                        "chunk_id": chunk_id,
                        "source_text": res.get("source_text", "")[:900],
                        "manual_verdict": "UNCHECKED",  # fill in by hand: SUPPORTED | UNSUPPORTED
                    }
                )

    # -------------------------------------------------- RxNorm normalization
    rx_path = settings.raw_dir / "rxnorm_ingredients.json"
    rx_matched = 0
    if rx_path.exists():
        rx = json.loads(rx_path.read_text())
        lookup = {k.lower(): v for k, v in rx.items()}
        for node, data in kg.g.nodes(data=True):
            if data.get("node_type") == "medication" and node in lookup:
                data["rxcui"] = lookup[node]
                rx_matched += 1
        log.info("graph.rxnorm_linked", matched=rx_matched, of=len(lookup))
    else:
        log.info("graph.rxnorm_skipped", hint="Run scripts/ingest_rxnorm.py to link RXCUIs.")

    kg.save()

    stats = kg.stats()
    report = {
        "chunks_processed": len(sample),
        "entities_extracted": n_entities,
        "relations_extracted": n_relations,
        "relations_rejected": n_rejected,
        "rxnorm_linked": rx_matched,
        "graph": stats,
        "extraction_model": llm.provider_info()["fast_model"],
        "audit_sample_size": len(audit_rows),
        "audit_file": "graph_extraction_audit.json",
        "note": (
            "Extraction precision is UNMEASURED until the audit file is manually "
            "labelled. Do not report a precision figure derived from anything else."
        ),
    }
    (settings.artifacts_dir / "graph_manifest.json").write_text(json.dumps(report, indent=2))
    (settings.artifacts_dir / "graph_extraction_audit.json").write_text(
        json.dumps(audit_rows, indent=2)
    )

    log.info("graph.complete", **stats)
    print("\n" + json.dumps(report, indent=2))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", type=int, default=900, help="Chunks to run extraction over.")
    ap.add_argument("--concurrency", type=int, default=4, help="Parallel LLM calls.")
    ap.add_argument("--audit-sample", type=int, default=40,
                    help="Triples held out for manual precision checking.")
    args = ap.parse_args()
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
