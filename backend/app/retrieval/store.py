"""Vector store abstraction with two interchangeable backends.

`PgVectorStore` is the real one: Neon Postgres + pgvector, HNSW index, cosine.
`LocalVectorStore` is a NumPy matrix on disk searched by brute force.

The fallback is not a token gesture. At our corpus scale (~40-60k chunks x 1024
dims, roughly 240 MB in float32) an exhaustive search is a single matmul that
returns in a handful of milliseconds — faster, in fact, than a network round trip
to a cloud Postgres. What pgvector buys is a scaling ceiling and real SQL
filtering, not demo-scale latency. Saying so plainly is more useful than pretending
the managed database is doing something magical at this size.

`get_store()` picks the backend from whether DATABASE_URL is set, and the choice is
logged and exposed on /health so it is never silent.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np

from app.config import settings
from app.logging_conf import get_logger
from app.retrieval.types import Chunk, SearchHit

log = get_logger(__name__)


@runtime_checkable
class VectorStore(Protocol):
    backend: str

    def upsert(self, chunks: list[Chunk], vectors: np.ndarray) -> int: ...
    def search(self, query_vector: np.ndarray, k: int) -> list[SearchHit]: ...
    def count(self) -> int: ...
    def get_parent(self, parent_id: str) -> str | None: ...
    def health(self) -> dict[str, object]: ...


# --------------------------------------------------------------------- local


class LocalVectorStore:
    """Brute-force cosine over a normalized float32 matrix held in memory."""

    backend = "local-numpy"

    def __init__(self, path: Path | None = None) -> None:
        self.path = path or settings.local_vectors_path
        self._vectors: np.ndarray = np.zeros((0, settings.embedding_dim), dtype=np.float32)
        self._chunks: list[Chunk] = []
        self._parents: dict[str, str] = {}
        if self.path.exists():
            self.load()

    # -- persistence -------------------------------------------------------

    def load(self) -> None:
        import pickle

        data = np.load(self.path, allow_pickle=False)
        self._vectors = data["vectors"].astype(np.float32)
        meta_path = self.path.with_suffix(".meta.pkl")
        if meta_path.exists():
            with meta_path.open("rb") as fh:
                blob = pickle.load(fh)
            self._chunks = blob["chunks"]
            self._parents = blob["parents"]
        log.info("store.local_loaded", vectors=len(self._vectors), chunks=len(self._chunks))

    def save(self) -> None:
        import pickle

        self.path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(self.path, vectors=self._vectors)
        with self.path.with_suffix(".meta.pkl").open("wb") as fh:
            pickle.dump({"chunks": self._chunks, "parents": self._parents}, fh)
        log.info("store.local_saved", path=str(self.path), vectors=len(self._vectors))

    def set_parents(self, parents: dict[str, str]) -> None:
        self._parents = parents

    # -- interface ---------------------------------------------------------

    def upsert(self, chunks: list[Chunk], vectors: np.ndarray) -> int:
        if len(chunks) != len(vectors):
            raise ValueError(f"chunk/vector length mismatch: {len(chunks)} vs {len(vectors)}")
        self._chunks.extend(chunks)
        self._vectors = (
            vectors.astype(np.float32)
            if len(self._vectors) == 0
            else np.vstack([self._vectors, vectors.astype(np.float32)])
        )
        return len(chunks)

    def search(self, query_vector: np.ndarray, k: int) -> list[SearchHit]:
        if len(self._vectors) == 0:
            return []
        # Both sides are L2-normalized at encode time, so the dot product IS cosine.
        sims = self._vectors @ query_vector.astype(np.float32)
        k = min(k, len(sims))
        # argpartition avoids a full sort of the whole corpus for a top-k slice.
        idx = np.argpartition(-sims, k - 1)[:k]
        idx = idx[np.argsort(-sims[idx])]
        return [
            SearchHit(chunk=self._chunks[i], score=float(sims[i]), retriever="dense", rank=r)
            for r, i in enumerate(idx)
        ]

    def count(self) -> int:
        return len(self._chunks)

    def get_parent(self, parent_id: str) -> str | None:
        return self._parents.get(parent_id)

    def health(self) -> dict[str, object]:
        return {
            "backend": self.backend,
            "chunks": self.count(),
            "parents": len(self._parents),
            "ready": self.count() > 0,
            "note": "Exhaustive cosine search. Set DATABASE_URL to use pgvector.",
        }


# --------------------------------------------------------------------- pgvector


DDL = """
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS parents (
    parent_id TEXT PRIMARY KEY,
    doc_id    TEXT NOT NULL,
    title     TEXT,
    source    TEXT,
    section   TEXT,
    text      TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS chunks (
    chunk_id    TEXT PRIMARY KEY,
    parent_id   TEXT REFERENCES parents(parent_id) ON DELETE CASCADE,
    doc_id      TEXT NOT NULL,
    title       TEXT,
    source      TEXT,
    section     TEXT,
    url         TEXT,
    token_count INTEGER DEFAULT 0,
    text        TEXT NOT NULL,
    embedding   vector(%(dim)s)
);

-- HNSW with vector_ip_ops because embeddings are L2-normalized, which makes inner
-- product order-equivalent to cosine while skipping the norm computation per probe.
CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw
    ON chunks USING hnsw (embedding vector_ip_ops)
    WITH (m = 16, ef_construction = 64);

CREATE INDEX IF NOT EXISTS chunks_doc_id_idx ON chunks (doc_id);

-- Postgres full-text, kept alongside the Python BM25 index so the sparse path has
-- a server-side option when the corpus outgrows in-process rank_bm25.
CREATE INDEX IF NOT EXISTS chunks_fts_idx
    ON chunks USING GIN (to_tsvector('english', text));

CREATE TABLE IF NOT EXISTS audit_log (
    id           BIGSERIAL PRIMARY KEY,
    run_id       TEXT NOT NULL,
    ts           TIMESTAMPTZ NOT NULL DEFAULT now(),
    node         TEXT NOT NULL,
    event        TEXT NOT NULL,
    duration_ms  DOUBLE PRECISION,
    payload      JSONB
);
CREATE INDEX IF NOT EXISTS audit_run_idx ON audit_log (run_id, ts);
"""


class PgVectorStore:
    """Neon/Postgres + pgvector."""

    backend = "pgvector"

    def __init__(self, dsn: str | None = None) -> None:
        self.dsn = dsn or settings.database_url
        if not self.dsn:
            raise ValueError("PgVectorStore requires DATABASE_URL")
        self._pool = None

    @property
    def pool(self):
        if self._pool is None:
            from psycopg_pool import ConnectionPool

            self._pool = ConnectionPool(
                self.dsn,
                min_size=settings.db_pool_min,
                max_size=settings.db_pool_max,
                open=True,
                kwargs={"autocommit": True},
            )
        return self._pool

    def init_schema(self) -> None:
        with self.pool.connection() as conn:
            conn.execute(DDL % {"dim": settings.embedding_dim})
        log.info("store.pg_schema_ready", dim=settings.embedding_dim)

    def upsert_parents(self, parents: list[tuple[str, str, str, str, str, str]]) -> int:
        with self.pool.connection() as conn, conn.cursor() as cur:
            cur.executemany(
                """INSERT INTO parents (parent_id, doc_id, title, source, section, text)
                   VALUES (%s,%s,%s,%s,%s,%s)
                   ON CONFLICT (parent_id) DO UPDATE SET text = EXCLUDED.text""",
                parents,
            )
        return len(parents)

    def upsert(self, chunks: list[Chunk], vectors: np.ndarray) -> int:
        from pgvector.psycopg import register_vector

        if len(chunks) != len(vectors):
            raise ValueError(f"chunk/vector length mismatch: {len(chunks)} vs {len(vectors)}")

        rows = [
            (
                c.chunk_id, c.parent_id, c.doc_id, c.title, c.source,
                c.section, c.url, c.token_count, c.text, v,
            )
            for c, v in zip(chunks, vectors, strict=True)
        ]
        with self.pool.connection() as conn:
            register_vector(conn)
            with conn.cursor() as cur:
                cur.executemany(
                    """INSERT INTO chunks
                       (chunk_id,parent_id,doc_id,title,source,section,url,token_count,text,embedding)
                       VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                       ON CONFLICT (chunk_id) DO UPDATE
                       SET embedding = EXCLUDED.embedding, text = EXCLUDED.text""",
                    rows,
                )
        return len(rows)

    def search(self, query_vector: np.ndarray, k: int) -> list[SearchHit]:
        from pgvector.psycopg import register_vector

        with self.pool.connection() as conn:
            register_vector(conn)
            with conn.cursor() as cur:
                # <#> is negative inner product; negate to recover cosine similarity.
                cur.execute(
                    """SELECT chunk_id,parent_id,doc_id,title,source,section,url,token_count,text,
                              -(embedding <#> %s) AS score
                       FROM chunks ORDER BY embedding <#> %s LIMIT %s""",
                    (query_vector, query_vector, k),
                )
                rows = cur.fetchall()

        return [
            SearchHit(
                chunk=Chunk(
                    chunk_id=r[0], parent_id=r[1], doc_id=r[2], title=r[3] or "",
                    source=r[4] or "", section=r[5] or "", url=r[6] or "",
                    token_count=r[7] or 0, text=r[8],
                ),
                score=float(r[9]),
                retriever="dense",
                rank=i,
            )
            for i, r in enumerate(rows)
        ]

    def count(self) -> int:
        try:
            with self.pool.connection() as conn, conn.cursor() as cur:
                cur.execute("SELECT count(*) FROM chunks")
                row = cur.fetchone()
                return int(row[0]) if row else 0
        except Exception:  # noqa: BLE001 — health check must not raise
            return 0

    def get_parent(self, parent_id: str) -> str | None:
        with self.pool.connection() as conn, conn.cursor() as cur:
            cur.execute("SELECT text FROM parents WHERE parent_id = %s", (parent_id,))
            row = cur.fetchone()
            return row[0] if row else None

    def all_chunks(self) -> list[Chunk]:
        """Used to build the in-process BM25 index from whatever pgvector holds."""
        with self.pool.connection() as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT chunk_id,parent_id,doc_id,title,source,section,url,token_count,text FROM chunks"
            )
            return [
                Chunk(
                    chunk_id=r[0], parent_id=r[1], doc_id=r[2], title=r[3] or "",
                    source=r[4] or "", section=r[5] or "", url=r[6] or "",
                    token_count=r[7] or 0, text=r[8],
                )
                for r in cur.fetchall()
            ]

    def health(self) -> dict[str, object]:
        n = self.count()
        return {"backend": self.backend, "chunks": n, "ready": n > 0}


# --------------------------------------------------------------------- factory

_store: VectorStore | None = None


def get_store(force_local: bool = False) -> VectorStore:
    global _store
    if _store is not None:
        return _store

    if settings.database_url and not force_local:
        try:
            pg = PgVectorStore()
            pg.init_schema()
            log.info("store.selected", backend="pgvector")
            _store = pg
            return _store
        except Exception as exc:  # noqa: BLE001
            # A cloud DB that is unreachable should degrade to a working demo, not a
            # 500 on every request — but it must say so.
            log.error("store.pg_unavailable", error=str(exc)[:250], falling_back="local-numpy")

    log.info("store.selected", backend="local-numpy")
    _store = LocalVectorStore()
    return _store


def reset_store() -> None:
    """Test hook."""
    global _store
    _store = None
