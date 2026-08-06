"""Shared retrieval types.

Kept in one small module so the vector stores, the BM25 index, the fusion layer and
the agent nodes all agree on shape without importing each other.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class Chunk:
    """A child chunk: the unit we retrieve on.

    `parent_id` links back to the larger passage handed to the composer. Retrieving
    on small chunks and composing on large ones is the whole point of parent-child
    chunking — precision where we search, context where we write.
    """

    chunk_id: str
    parent_id: str
    doc_id: str
    text: str
    title: str = ""
    source: str = ""
    section: str = ""
    url: str = ""
    token_count: int = 0
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Parent:
    parent_id: str
    doc_id: str
    text: str
    title: str = ""
    source: str = ""
    section: str = ""


@dataclass(slots=True)
class SearchHit:
    """A retrieved chunk plus how it was found.

    `retriever` and `rank` are carried through fusion so the audit log can say which
    path surfaced each piece of evidence — 'the graph found this, not the vector
    search' is exactly the kind of claim the governance layer needs to support.
    """

    chunk: Chunk
    score: float
    retriever: str  # "dense" | "bm25" | "graph" | "fused"
    rank: int = 0
    components: dict[str, float] = field(default_factory=dict)

    def to_citation(self, index: int) -> dict[str, Any]:
        c = self.chunk
        return {
            "index": index,
            "chunk_id": c.chunk_id,
            "parent_id": c.parent_id,
            "title": c.title or c.doc_id,
            "source": c.source,
            "section": c.section,
            "url": c.url,
            "score": round(self.score, 4),
            "retriever": self.retriever,
            "snippet": c.text[:320] + ("…" if len(c.text) > 320 else ""),
        }
