"""BM25 sparse retrieval.

Dense retrieval is weak exactly where clinical text is most demanding: rare drug
names, dosages, ICD codes, and abbreviations that the embedding model has smoothed
into a neighbourhood of similar-looking tokens. BM25 matches those literally. The
two failure modes are close to complementary, which is why the hybrid is worth the
extra index rather than being cargo-culted.
"""

from __future__ import annotations

import pickle
import re
from pathlib import Path

from app.config import settings
from app.logging_conf import get_logger
from app.retrieval.types import Chunk, SearchHit

log = get_logger(__name__)

_TOKEN = re.compile(r"[a-z0-9]+(?:[-/][a-z0-9]+)*")

# Kept deliberately short. Aggressive stoplists strip clinically load-bearing words
# ("no", "not", "without" invert a finding), so we only drop true function words.
_STOP = frozenset(
    ["a", "an", "the", "and", "or", "of", "to", "in", "for", "on", "at", "by", "with", "from", "as", "is", "are", "was", "were", "be", "been", "being", "this", "that", "these", "those", "it", "its", "which", "what", "when", "where", "who", "whom", "how"]
)


def tokenize(text: str) -> list[str]:
    """Lowercase word tokens, keeping hyphens and slashes.

    'piperacillin/tazobactam' and 'beta-blocker' are single clinical concepts; a
    tokenizer that splits them turns one precise term into two vague ones.
    """
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOP and len(t) > 1]


class BM25Index:
    def __init__(self) -> None:
        self._bm25 = None
        self._chunks: list[Chunk] = []

    def build(self, chunks: list[Chunk]) -> None:
        from rank_bm25 import BM25Okapi

        self._chunks = chunks
        corpus = [tokenize(f"{c.title} {c.section} {c.text}") for c in chunks]
        self._bm25 = BM25Okapi(corpus)
        log.info("bm25.built", chunks=len(chunks))

    def search(self, query: str, k: int) -> list[SearchHit]:
        if self._bm25 is None or not self._chunks:
            return []
        tokens = tokenize(query)
        if not tokens:
            return []

        scores = self._bm25.get_scores(tokens)
        k = min(k, len(scores))
        top = sorted(range(len(scores)), key=lambda i: -scores[i])[:k]
        return [
            SearchHit(chunk=self._chunks[i], score=float(scores[i]), retriever="bm25", rank=r)
            for r, i in enumerate(top)
            if scores[i] > 0
        ]

    def save(self, path: Path | None = None) -> None:
        p = path or settings.bm25_path
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("wb") as fh:
            pickle.dump({"bm25": self._bm25, "chunks": self._chunks}, fh)
        log.info("bm25.saved", path=str(p))

    @classmethod
    def load(cls, path: Path | None = None) -> BM25Index | None:
        p = path or settings.bm25_path
        if not p.exists():
            return None
        idx = cls()
        with p.open("rb") as fh:
            blob = pickle.load(fh)
        idx._bm25 = blob["bm25"]
        idx._chunks = blob["chunks"]
        log.info("bm25.loaded", chunks=len(idx._chunks))
        return idx

    def count(self) -> int:
        return len(self._chunks)


_index: BM25Index | None = None


def get_bm25() -> BM25Index | None:
    global _index
    if _index is None:
        _index = BM25Index.load()
    return _index


def reset_bm25() -> None:
    global _index
    _index = None
