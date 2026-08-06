"""Parent-child chunking.

We retrieve on small chunks and compose on large ones.

A ~450-token child is small enough that its embedding represents one idea rather
than an average of several, which is what makes dense retrieval precise. But a
passage that size is often too narrow to answer from — it may state a dose without
the indication, or a contraindication without the drug. So each child carries a
`parent_id`, and once retrieval has selected children, the composer is handed the
~2000-token parents they came from.

The alternative — retrieving on large chunks — degrades embedding precision, because
a single vector must represent a passage covering several distinct claims. The
alternative of retrieving on small chunks and composing on them too produces
confidently wrong answers assembled from fragments. Parent-child avoids both.

Splitting is sentence-aware: a chunk boundary mid-sentence can sever a negation
("... is **not** indicated in patients with ...") from what it negates, which in this
domain inverts the meaning rather than merely blurring it.
"""

from __future__ import annotations

import hashlib
import re

from app.config import settings
from app.retrieval.types import Chunk, Parent

# Sentence boundary: period/question/exclamation + space + capital, while protecting
# the abbreviations that are dense in clinical text (mg., i.e., e.g., Dr., vs.).
_ABBREV = r"(?<!\bDr)(?<!\bvs)(?<!\bi\.e)(?<!\be\.g)(?<!\bmg)(?<!\bml)(?<!\bkg)(?<!\bNo)(?<!\bFig)"
_SENT_SPLIT = re.compile(rf"{_ABBREV}(?<=[.!?])\s+(?=[A-Z0-9])")

_HEADING = re.compile(r"^\s{0,3}(#{1,6}\s+.+|[A-Z][A-Z0-9 \-/&,()]{6,80})\s*$", re.M)


def count_tokens(text: str) -> int:
    """Approximate token count.

    Deliberately not a real tokenizer call. This runs tens of thousands of times
    during ingestion and the exact figure only needs to be good enough to keep child
    chunks under the model's 512-token limit. Whitespace words x 1.35 tracks
    subword tokenizers closely enough on English prose; we keep the safety margin in
    the configured chunk size, not in the estimator's precision.
    """
    return int(len(text.split()) * 1.35)


def _stable_id(prefix: str, *parts: str) -> str:
    h = hashlib.sha1("||".join(parts).encode("utf-8")).hexdigest()[:16]
    return f"{prefix}_{h}"


def split_sentences(text: str) -> list[str]:
    parts = [s.strip() for s in _SENT_SPLIT.split(text) if s.strip()]
    return parts or ([text.strip()] if text.strip() else [])


def detect_section(text: str) -> str:
    m = _HEADING.search(text)
    if not m:
        return ""
    return m.group(1).lstrip("#").strip()[:120]


def _pack(units: list[str], max_tokens: int, overlap_tokens: int = 0) -> list[str]:
    """Greedily pack units into blocks under max_tokens, with optional overlap."""
    blocks: list[str] = []
    current: list[str] = []
    current_tokens = 0

    for unit in units:
        ut = count_tokens(unit)
        # A single oversized unit (a huge table or an unsplittable paragraph) gets
        # hard-wrapped on word count rather than silently dropped.
        if ut > max_tokens:
            if current:
                blocks.append(" ".join(current))
                current, current_tokens = [], 0
            words = unit.split()
            stride = int(max_tokens / 1.35)
            for i in range(0, len(words), stride):
                blocks.append(" ".join(words[i : i + stride]))
            continue

        if current_tokens + ut > max_tokens and current:
            blocks.append(" ".join(current))
            if overlap_tokens > 0:
                # Carry trailing sentences forward so a claim split across a boundary
                # still appears whole in one of the two chunks.
                carry: list[str] = []
                carried = 0
                for prev in reversed(current):
                    pt = count_tokens(prev)
                    if carried + pt > overlap_tokens:
                        break
                    carry.insert(0, prev)
                    carried += pt
                current, current_tokens = carry, carried
            else:
                current, current_tokens = [], 0

        current.append(unit)
        current_tokens += ut

    if current:
        blocks.append(" ".join(current))
    return [b for b in blocks if b.strip()]


def chunk_document(
    *,
    doc_id: str,
    text: str,
    title: str = "",
    source: str = "",
    url: str = "",
) -> tuple[list[Parent], list[Chunk]]:
    """Split one document into parents and their child chunks."""
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if not text:
        return [], []

    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    parent_texts = _pack(paragraphs, settings.parent_chunk_tokens)

    parents: list[Parent] = []
    children: list[Chunk] = []

    for p_i, ptext in enumerate(parent_texts):
        parent_id = _stable_id("p", doc_id, str(p_i))
        section = detect_section(ptext)
        parents.append(
            Parent(
                parent_id=parent_id, doc_id=doc_id, text=ptext,
                title=title, source=source, section=section,
            )
        )

        sentences = split_sentences(ptext)
        child_texts = _pack(
            sentences, settings.child_chunk_tokens, settings.child_chunk_overlap
        )
        for c_i, ctext in enumerate(child_texts):
            children.append(
                Chunk(
                    chunk_id=_stable_id("c", doc_id, str(p_i), str(c_i)),
                    parent_id=parent_id,
                    doc_id=doc_id,
                    text=ctext,
                    title=title,
                    source=source,
                    section=section,
                    url=url,
                    token_count=count_tokens(ctext),
                )
            )

    return parents, children
