"""Parent-child chunking invariants.

The properties tested here are the ones that silently break retrieval when they
regress: child chunks exceeding the embedding model's context (silent truncation,
so the tail of every long chunk stops being searchable), and orphaned parent links
(the composer falls back to narrow context without any error).
"""

from __future__ import annotations

from app.config import settings
from app.retrieval.chunking import chunk_document, count_tokens, split_sentences

LONG_DOC = (
    "Sepsis is a life-threatening organ dysfunction caused by a dysregulated host "
    "response to infection. Early recognition is essential. "
) * 60


def test_children_stay_within_embedding_context():
    """Every child must fit bge-large's 512-token window.

    A chunk over the limit is truncated by the encoder without raising, so its tail
    becomes permanently unsearchable — a silent recall hole, not a crash.
    """
    _, children = chunk_document(doc_id="d1", text=LONG_DOC, title="Sepsis")
    assert children
    for c in children:
        assert c.token_count <= 512, f"chunk {c.chunk_id} has {c.token_count} tokens"


def test_children_respect_configured_size():
    _, children = chunk_document(doc_id="d1", text=LONG_DOC, title="Sepsis")
    # Allow one sentence of overshoot: packing is sentence-atomic, so the last
    # sentence added can carry the block slightly past the target.
    limit = settings.child_chunk_tokens + 120
    assert all(c.token_count <= limit for c in children)


def test_every_child_links_to_a_real_parent():
    parents, children = chunk_document(doc_id="d1", text=LONG_DOC, title="Sepsis")
    parent_ids = {p.parent_id for p in parents}
    assert children
    for c in children:
        assert c.parent_id in parent_ids


def test_ids_are_stable_across_runs():
    """Re-ingesting the same document must not duplicate rows in the index."""
    a_parents, a_children = chunk_document(doc_id="d1", text=LONG_DOC, title="Sepsis")
    b_parents, b_children = chunk_document(doc_id="d1", text=LONG_DOC, title="Sepsis")
    assert [p.parent_id for p in a_parents] == [p.parent_id for p in b_parents]
    assert [c.chunk_id for c in a_children] == [c.chunk_id for c in b_children]


def test_different_documents_get_different_ids():
    _, a = chunk_document(doc_id="d1", text=LONG_DOC)
    _, b = chunk_document(doc_id="d2", text=LONG_DOC)
    assert {c.chunk_id for c in a}.isdisjoint({c.chunk_id for c in b})


def test_empty_input_is_handled():
    assert chunk_document(doc_id="d", text="") == ([], [])
    assert chunk_document(doc_id="d", text="   \n\n  ") == ([], [])


def test_oversized_single_paragraph_is_wrapped_not_dropped():
    """An unsplittable wall of text must still be indexed, not silently discarded."""
    wall = "word " * 5000
    _, children = chunk_document(doc_id="d", text=wall)
    assert children
    assert all(c.token_count <= 512 for c in children)


def test_sentence_split_protects_clinical_abbreviations():
    """Splitting on 'mg.' or 'e.g.' severs doses and qualifiers from their context."""
    text = "Give 500 mg. of the drug orally. Repeat in 6 hours."
    sentences = split_sentences(text)
    assert not any(s.strip() == "of the drug orally." for s in sentences)


def test_token_estimate_is_conservative():
    """The estimator may overshoot; undershooting would let chunks exceed the window."""
    text = "the quick brown fox jumps over the lazy dog"
    assert count_tokens(text) >= len(text.split())
