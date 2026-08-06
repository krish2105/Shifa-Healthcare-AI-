#!/usr/bin/env python
"""Phase A.1 — ingest the clinical guideline corpus.

Source: `epfl-llm/guidelines` (Hugging Face). Public, ungated, verified 2026-08-06.
Columns: id, source, title, clean_text, raw_text, url, overview.

Two decisions worth stating, because both affect every number downstream.

**Stratified sampling, not head-N.** The corpus is ordered by source, so taking the
first 2,000 rows yields a corpus that is almost entirely one publisher's oncology
guidance. Retrieval would then look excellent on cancer questions and fall apart on
everything else — and the benchmark average would hide it. We cap per source and
sample across all of them, so coverage failures show up in the metrics instead of
being designed out of the sample.

**Measured counts only.** Everything this script learns about the corpus is written
to `ingest_manifest.json`. No document, chunk, or source count appears in the README
or RESULTS.md that did not come from that file.

Usage:
    python scripts/ingest_guidelines.py                # default limit from settings
    python scripts/ingest_guidelines.py --limit 500    # quick pass
    python scripts/ingest_guidelines.py --max-scan 50000
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from app.config import settings
from app.logging_conf import get_logger
from app.retrieval.chunking import chunk_document

log = get_logger("ingest.guidelines")

DATASET = "epfl-llm/guidelines"

# Boilerplate that survives the corpus's own cleaning and adds no clinical signal.
_JUNK_MARKERS = (
    "javascript is disabled",
    "your browser is not supported",
    "cookies on this site",
    "404 not found",
    "access denied",
)


def is_usable(text: str, title: str) -> tuple[bool, str]:
    """Quality gate. Returns (keep, reason_if_rejected)."""
    if not text or not text.strip():
        return False, "empty"
    stripped = text.strip()
    if len(stripped) < settings.min_guideline_chars:
        return False, "too_short"
    low = stripped[:3000].lower()
    if any(m in low for m in _JUNK_MARKERS):
        return False, "boilerplate"
    # Guideline prose is mostly words; a page that is mostly punctuation and digits
    # is a table dump or a reference list, which retrieves badly and cites worse.
    alpha = sum(c.isalpha() or c.isspace() for c in stripped[:4000])
    if alpha / min(len(stripped), 4000) < 0.75:
        return False, "low_text_ratio"
    return True, ""


def allocate_quotas(available: dict[str, int], limit: int) -> dict[str, int]:
    """Spread `limit` documents across sources as evenly as their supply allows.

    Equal shares first; whatever a small source cannot fill is redistributed to the
    sources that still have documents left. The result is as close to uniform
    coverage as the corpus permits, without silently dropping a small publisher just
    because a large one could fill the whole budget on its own.
    """
    quotas: dict[str, int] = dict.fromkeys(available, 0)
    remaining = limit
    active = {s for s, n in available.items() if n > 0}

    while remaining > 0 and active:
        share = max(1, remaining // len(active))
        progressed = False
        for src in sorted(active, key=lambda s: available[s]):
            if remaining <= 0:
                break
            take = min(share, available[src] - quotas[src], remaining)
            if take <= 0:
                continue
            quotas[src] += take
            remaining -= take
            progressed = True
        active = {s for s in active if quotas[s] < available[s]}
        if not progressed:
            break

    return {s: q for s, q in quotas.items() if q > 0}


def main() -> int:
    ap = argparse.ArgumentParser(description="Ingest clinical guidelines into parent/child chunks.")
    ap.add_argument("--limit", type=int, default=settings.guideline_doc_limit,
                    help="Target number of documents to keep.")
    ap.add_argument("--max-scan", type=int, default=0,
                    help="Cap rows examined. 0 = whole corpus.")
    args = ap.parse_args()

    settings.ensure_dirs()

    from datasets import load_dataset

    log.info("ingest.start", dataset=DATASET, target_docs=args.limit)

    # Non-streaming on purpose. The corpus is source-ordered, so any single-pass
    # streaming scan sees only one publisher until deep into the data — stratifying
    # correctly requires knowing the full source distribution up front. Downloading
    # once also makes every subsequent run instant off the HF cache.
    ds = load_dataset(DATASET, split="train")
    total_rows = len(ds) if args.max_scan == 0 else min(len(ds), args.max_scan)
    log.info("ingest.loaded", total_rows=len(ds), scanning=total_rows)

    # -- pass 1: which documents pass the quality gate, grouped by source ------
    eligible: dict[str, list[int]] = defaultdict(list)
    rejected: Counter[str] = Counter()

    texts = ds["clean_text"]
    raw_texts = ds["raw_text"] if "raw_text" in ds.column_names else None
    sources = ds["source"]
    titles = ds["title"]

    for i in range(total_rows):
        text = (texts[i] or (raw_texts[i] if raw_texts else "") or "").strip()
        ok, reason = is_usable(text, titles[i] or "")
        if not ok:
            rejected[reason] += 1
            continue
        eligible[(sources[i] or "unknown").strip()].append(i)

    available = {s: len(idxs) for s, idxs in eligible.items()}
    log.info("ingest.eligible", sources=len(available), total=sum(available.values()),
             rejected=dict(rejected))

    # -- quota allocation ------------------------------------------------------
    quotas = allocate_quotas(available, args.limit)
    log.info("ingest.quotas", quotas=dict(sorted(quotas.items(), key=lambda kv: -kv[1])))

    # -- pass 2: take an evenly-spread slice from each source ------------------
    kept: list[dict] = []
    per_source: Counter[str] = Counter()

    for src, quota in quotas.items():
        idxs = eligible[src]
        # Even stride rather than the first N: documents within a source are often
        # ordered by topic, so head-N would over-sample one clinical area again.
        stride = max(1, len(idxs) // quota)
        chosen = idxs[::stride][:quota]

        for i in chosen:
            text = (texts[i] or (raw_texts[i] if raw_texts else "") or "").strip()
            title = (titles[i] or "").strip()
            if title.lower() in ("none", "null"):
                title = ""
            url = (ds[i].get("url") or "").strip()
            kept.append(
                {
                    "doc_id": str(ds[i].get("id") or f"{src}_{i}"),
                    "source": src,
                    "title": title or f"{src} guideline {per_source[src] + 1}",
                    "text": text,
                    "url": "" if url.lower() in ("none", "null") else url,
                }
            )
            per_source[src] += 1

    scanned = total_rows
    log.info(
        "ingest.selection_done",
        kept=len(kept), scanned=scanned,
        sources=len(per_source), rejected=dict(rejected),
    )

    if not kept:
        log.error("ingest.no_documents", msg="Nothing passed the quality gate.")
        return 1

    # ---------------------------------------------------------------- chunking
    all_parents: list[dict] = []
    all_chunks: list[dict] = []

    for i, doc in enumerate(kept):
        parents, children = chunk_document(
            doc_id=doc["doc_id"], text=doc["text"],
            title=doc["title"], source=doc["source"], url=doc["url"],
        )
        all_parents.extend(
            {
                "parent_id": p.parent_id, "doc_id": p.doc_id, "text": p.text,
                "title": p.title, "source": p.source, "section": p.section,
            }
            for p in parents
        )
        all_chunks.extend(
            {
                "chunk_id": c.chunk_id, "parent_id": c.parent_id, "doc_id": c.doc_id,
                "text": c.text, "title": c.title, "source": c.source,
                "section": c.section, "url": c.url, "token_count": c.token_count,
            }
            for c in children
        )
        if (i + 1) % 250 == 0:
            log.info("chunk.progress", docs=i + 1, chunks=len(all_chunks))

    parents_df = pd.DataFrame(all_parents)
    chunks_df = pd.DataFrame(all_chunks)

    parents_df.to_parquet(settings.parents_path, index=False)
    chunks_df.to_parquet(settings.chunks_path, index=False)

    tok = chunks_df["token_count"]
    doc_chars = [len(d["text"]) for d in kept]

    manifest = {
        "dataset": DATASET,
        "rows_scanned": scanned,
        "documents_kept": len(kept),
        "documents_rejected": dict(rejected),
        "sources": dict(per_source.most_common()),
        "n_sources": len(per_source),
        "parent_chunks": len(all_parents),
        "child_chunks": len(all_chunks),
        "child_tokens": {
            "mean": round(float(tok.mean()), 1),
            "median": float(tok.median()),
            "p95": float(tok.quantile(0.95)),
            "max": int(tok.max()),
        },
        "doc_chars": {
            "mean": round(sum(doc_chars) / len(doc_chars), 1),
            "min": min(doc_chars),
            "max": max(doc_chars),
        },
        "config": {
            "child_chunk_tokens": settings.child_chunk_tokens,
            "child_chunk_overlap": settings.child_chunk_overlap,
            "parent_chunk_tokens": settings.parent_chunk_tokens,
            "min_guideline_chars": settings.min_guideline_chars,
        },
    }
    settings.ingest_manifest_path.write_text(json.dumps(manifest, indent=2))

    log.info(
        "ingest.complete",
        documents=len(kept), parents=len(all_parents), chunks=len(all_chunks),
        sources=len(per_source),
    )
    print("\n" + json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
