#!/usr/bin/env python
"""Phase B — embed child chunks and build the hybrid index.

Produces:
  * dense vectors, written to pgvector when DATABASE_URL is set, otherwise to a
    local .npz (see app/retrieval/store.py for why the fallback is real, not a stub)
  * a BM25 sparse index over the same chunks
  * parent passages, so the composer can be handed context wider than what was
    retrieved on

Embeddings are computed locally with bge-large-en-v1.5. No API cost, and no corpus
text leaves the machine — which for a project whose thesis is data governance is
worth more than the money saved.

**Crash safety.** This is a 45-90 minute GPU job, and on Apple Silicon the MPS
backend can fail mid-run with an internal Metal command-buffer error (observed here
at ~30k chunks after ~37 minutes of sustained work). An all-or-nothing script loses
the entire run to that. So:

  * work is checkpointed to shards on disk after every shard, and `--resume`
    (default) skips shards already on disk;
  * a Metal/MPS failure triggers a cache flush and one retry, then permanently
    falls back to CPU for the remainder rather than aborting;
  * `torch.mps.empty_cache()` runs periodically, since the crash correlates with
    accumulated allocator pressure rather than any single batch.

Usage:
    python scripts/embed_index.py                  # resumes automatically
    python scripts/embed_index.py --restart        # discard shards, start over
    python scripts/embed_index.py --device cpu     # slower but stable
    python scripts/embed_index.py --limit 5000     # partial index for a fast loop
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from app.config import settings
from app.logging_conf import get_logger
from app.retrieval.bm25 import BM25Index
from app.retrieval.types import Chunk

log = get_logger("embed.index")

SHARD_DIR = settings.artifacts_dir / "embed_shards"
SHARD_SIZE = 2048


def _is_metal_error(exc: BaseException) -> bool:
    s = str(exc).lower()
    return any(k in s for k in ("metal", "mps", "command buffer", "internal error"))


class Encoder:
    """Wraps sentence-transformers with MPS failure recovery.

    Kept local to this script rather than folded into app/retrieval/embedder.py:
    the API path encodes one short query at a time and never triggers this, while
    the ingestion path runs the model flat out for an hour. Only the latter needs
    the recovery machinery, and the serving path should not pay for it.
    """

    def __init__(self, device: str) -> None:
        self.device = device
        self.model = None
        self._degraded_to_cpu = False
        self._load()

    def _load(self) -> None:
        from sentence_transformers import SentenceTransformer

        log.info("encoder.loading", model=settings.embedding_model, device=self.device)
        self.model = SentenceTransformer(settings.embedding_model, device=self.device)
        if settings.embedding_fp16 and self.device in ("mps", "cuda"):
            self.model = self.model.half()

    def _flush(self) -> None:
        try:
            import torch

            if self.device == "mps":
                torch.mps.empty_cache()
            elif self.device == "cuda":
                torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass

    def _fall_back_to_cpu(self) -> None:
        log.warning(
            "encoder.fallback_cpu",
            msg="MPS failed; continuing on CPU. Slower (~4.4 vs ~13 chunks/s) but stable.",
        )
        self.device = "cpu"
        self._degraded_to_cpu = True
        self._load()

    def encode(self, texts: list[str]) -> np.ndarray:
        for attempt in range(2):
            try:
                vecs = self.model.encode(
                    texts,
                    batch_size=settings.embedding_batch_size,
                    normalize_embeddings=True,
                    convert_to_numpy=True,
                    show_progress_bar=False,
                )
                return np.asarray(vecs, dtype=np.float32)
            except Exception as exc:
                if not _is_metal_error(exc) or self.device == "cpu":
                    raise
                log.warning("encoder.metal_error", attempt=attempt, error=str(exc)[:160])
                self._flush()
                if attempt == 0:
                    self._load()  # rebuild the graph; a wedged context does not recover
                else:
                    self._fall_back_to_cpu()
        return self.encode(texts)  # post-fallback retry, now on CPU

    def maybe_flush(self, shard_index: int) -> None:
        # Every few shards. The Metal failure tracks accumulated allocator
        # pressure, not any particular batch.
        if self.device == "mps" and shard_index % 4 == 3:
            self._flush()


def resolve_device(requested: str) -> str:
    if requested != "auto":
        return requested
    import torch

    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def shard_path(i: int) -> Path:
    return SHARD_DIR / f"shard_{i:05d}.npy"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="Cap chunks indexed. 0 = all.")
    ap.add_argument("--shard-size", type=int, default=SHARD_SIZE)
    ap.add_argument("--device", default="auto", choices=["auto", "mps", "cuda", "cpu"])
    ap.add_argument("--restart", action="store_true", help="Discard existing shards.")
    ap.add_argument("--force-local", action="store_true", help="Ignore DATABASE_URL.")
    ap.add_argument("--skip-bm25", action="store_true")
    args = ap.parse_args()

    settings.ensure_dirs()

    if not settings.chunks_path.exists():
        log.error("embed.no_chunks", path=str(settings.chunks_path),
                  hint="Run scripts/ingest_guidelines.py first.")
        return 1

    chunks_df = pd.read_parquet(settings.chunks_path)
    parents_df = pd.read_parquet(settings.parents_path)
    if args.limit:
        chunks_df = chunks_df.head(args.limit)

    chunks = [
        Chunk(
            chunk_id=r.chunk_id, parent_id=r.parent_id, doc_id=r.doc_id,
            text=r.text, title=r.title or "", source=r.source or "",
            section=r.section or "", url=r.url or "", token_count=int(r.token_count or 0),
        )
        for r in chunks_df.itertuples(index=False)
    ]

    if args.restart and SHARD_DIR.exists():
        shutil.rmtree(SHARD_DIR)
    SHARD_DIR.mkdir(parents=True, exist_ok=True)

    n_shards = (len(chunks) + args.shard_size - 1) // args.shard_size
    existing = sorted(SHARD_DIR.glob("shard_*.npy"))
    log.info("embed.start", chunks=len(chunks), shards=n_shards, already_done=len(existing))

    # ------------------------------------------------------------ encode
    device = resolve_device(args.device)
    encoder: Encoder | None = None
    t0 = time.perf_counter()
    encoded = 0

    for s in range(n_shards):
        path = shard_path(s)
        lo = s * args.shard_size
        hi = min(lo + args.shard_size, len(chunks))

        if path.exists():
            try:
                arr = np.load(path)
                if arr.shape[0] == hi - lo and arr.shape[1] == settings.embedding_dim:
                    continue
                log.warning("embed.shard_malformed", shard=s, shape=list(arr.shape))
            except Exception:  # noqa: BLE001
                log.warning("embed.shard_unreadable", shard=s)
            path.unlink(missing_ok=True)

        if encoder is None:
            encoder = Encoder(device)

        batch_texts = [c.text for c in chunks[lo:hi]]
        vecs = encoder.encode(batch_texts)

        # Write to a temp file then rename: an interrupted write must not leave a
        # truncated shard that a later resume would trust.
        #
        # Written through an open handle rather than by path — np.save appends
        # ".npy" to a path that lacks it, so `np.save(x.tmp)` silently produces
        # "x.tmp.npy" and the subsequent rename fails.
        tmp = path.with_suffix(".tmp")
        with tmp.open("wb") as fh:
            np.save(fh, vecs)
        tmp.replace(path)

        encoded += len(batch_texts)
        encoder.maybe_flush(s)

        elapsed = time.perf_counter() - t0
        rate = encoded / elapsed if elapsed else 0
        remaining = ((n_shards - s - 1) * args.shard_size) / rate if rate else 0
        log.info(
            "embed.progress",
            shard=f"{s + 1}/{n_shards}", done=hi, of=len(chunks),
            rate_per_s=round(rate, 1), eta_s=round(remaining), device=encoder.device,
        )

    # ------------------------------------------------------- assemble store
    log.info("embed.assembling", shards=n_shards)
    vectors = np.concatenate([np.load(shard_path(s)) for s in range(n_shards)], axis=0)
    if len(vectors) != len(chunks):
        log.error("embed.shard_mismatch", vectors=len(vectors), chunks=len(chunks))
        return 1

    from app.retrieval.store import LocalVectorStore, PgVectorStore

    use_pg = bool(settings.database_url) and not args.force_local
    store: PgVectorStore | LocalVectorStore

    if use_pg:
        store = PgVectorStore()
        store.init_schema()
        store.upsert_parents(
            [
                (r.parent_id, r.doc_id, r.title or "", r.source or "", r.section or "", r.text)
                for r in parents_df.itertuples(index=False)
            ]
        )
        # Chunked upserts: a single 53k-row executemany over the wire to a cloud
        # Postgres routinely trips statement timeouts.
        for i in range(0, len(chunks), 2000):
            store.upsert(chunks[i : i + 2000], vectors[i : i + 2000])
            log.info("embed.pg_upsert", done=min(i + 2000, len(chunks)), of=len(chunks))
        log.info("embed.backend", backend="pgvector")
    else:
        store = LocalVectorStore()
        store.set_parents({r.parent_id: r.text for r in parents_df.itertuples(index=False)})
        store.upsert(chunks, vectors)
        store.save()
        log.info("embed.backend", backend="local-numpy")

    elapsed = time.perf_counter() - t0

    # ---------------------------------------------------------- sparse index
    bm25_stats: dict[str, object] = {"built": False}
    if not args.skip_bm25:
        bm = BM25Index()
        bm.build(chunks)
        bm.save()
        bm25_stats = {"built": True, "chunks": bm.count()}

    # ---------------------------------------------------------------- verify
    from app.retrieval.embedder import embed_query

    probe = "first-line antibiotic treatment for community acquired pneumonia in adults"
    hits = store.search(embed_query(probe), 3)

    report = {
        "backend": store.backend,
        "chunks_indexed": len(chunks),
        "parents": len(parents_df),
        "embedding_model": settings.embedding_model,
        "embedding_dim": settings.embedding_dim,
        "device": encoder.device if encoder else "cached-shards",
        "fell_back_to_cpu": bool(encoder and encoder._degraded_to_cpu),  # noqa: SLF001
        "encode_elapsed_s": round(elapsed, 1),
        "chunks_encoded_this_run": encoded,
        "bm25": bm25_stats,
        "smoke_probe": {
            "query": probe,
            "results": [
                {"score": round(h.score, 4), "source": h.chunk.source,
                 "title": h.chunk.title[:70], "snippet": h.chunk.text[:140]}
                for h in hits
            ],
        },
    }
    (settings.processed_dir / "index_manifest.json").write_text(json.dumps(report, indent=2))

    log.info("embed.complete", chunks=len(chunks), backend=store.backend)
    print("\n" + json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
