"""Retrieval fusion and scoring tests."""

from __future__ import annotations

from app.eval.metrics import lexical_faithfulness, split_claims
from app.retrieval.bm25 import tokenize
from app.retrieval.fusion import deduplicate, mmr_select, reciprocal_rank_fusion
from app.retrieval.types import Chunk, SearchHit


def hit(cid: str, rank: int, retriever: str, parent: str = "", text: str = "text") -> SearchHit:
    return SearchHit(
        chunk=Chunk(chunk_id=cid, parent_id=parent or f"p_{cid}", doc_id="d", text=text),
        score=1.0 / (rank + 1),
        retriever=retriever,
        rank=rank,
    )


# ------------------------------------------------------------------ RRF


def test_rrf_rewards_agreement_between_retrievers():
    """A chunk both retrievers rank highly should beat one only a single retriever
    found — that is the entire reason for fusing rather than concatenating."""
    dense = [hit("a", 0, "dense"), hit("b", 1, "dense")]
    sparse = [hit("b", 0, "bm25"), hit("c", 1, "bm25")]

    fused = reciprocal_rank_fusion([dense, sparse])
    ids = [f.chunk.chunk_id for f in fused]
    assert ids[0] == "b"


def test_rrf_ignores_score_magnitudes():
    """BM25 scores are unbounded, cosine is [-1,1]. Fusion must be rank-only, or a
    single large BM25 score would dominate every result."""
    big = SearchHit(chunk=Chunk("x", "px", "d", "t"), score=9999.0, retriever="bm25", rank=1)
    small = SearchHit(chunk=Chunk("y", "py", "d", "t"), score=0.01, retriever="dense", rank=0)

    fused = reciprocal_rank_fusion([[small], [big]])
    assert fused[0].chunk.chunk_id == "y"  # better rank wins despite tiny score


def test_rrf_records_component_scores():
    fused = reciprocal_rank_fusion([[hit("a", 0, "dense")], [hit("a", 0, "bm25")]])
    assert set(fused[0].components) == {"dense", "bm25"}


def test_rrf_handles_empty_input():
    assert reciprocal_rank_fusion([]) == []


def test_rrf_weights_change_ordering():
    dense = [hit("a", 0, "dense")]
    graph = [hit("b", 0, "graph")]
    weighted = reciprocal_rank_fusion([dense, graph], weights=[1.0, 0.1])
    assert weighted[0].chunk.chunk_id == "a"


# ------------------------------------------------------------------ dedup


def test_dedup_collapses_chunks_sharing_a_parent():
    """Two children of one passage are one piece of evidence; counting both
    overstates how much support an answer has."""
    hits = [
        hit("c1", 0, "dense", parent="p1"),
        hit("c2", 1, "dense", parent="p1"),
        hit("c3", 2, "dense", parent="p2"),
    ]
    out = deduplicate(hits)
    assert [h.chunk.chunk_id for h in out] == ["c1", "c3"]


def test_dedup_keeps_the_highest_ranked_of_a_group():
    hits = [hit("c1", 0, "dense", parent="p1"), hit("c2", 1, "dense", parent="p1")]
    assert deduplicate(hits)[0].chunk.chunk_id == "c1"


# ------------------------------------------------------------------ MMR


def test_mmr_drops_near_duplicate_passages():
    """Guideline corpora restate the same recommendation constantly; five
    paraphrases is not five pieces of evidence."""
    same = "target mean arterial pressure of 65 mmHg within the first hour"
    hits = [
        hit("a", 0, "fused", text=same),
        hit("b", 1, "fused", text=same + " for adults"),
        hit("c", 2, "fused", text="administer vitamin K for warfarin reversal in bleeding"),
    ]
    selected = mmr_select(hits, None, k=2, lambda_mult=0.5)
    ids = {h.chunk.chunk_id for h in selected}
    assert "c" in ids, "MMR should surface the distinct passage over a paraphrase"


def test_mmr_returns_everything_when_k_covers_input():
    hits = [hit("a", 0, "fused"), hit("b", 1, "fused")]
    assert len(mmr_select(hits, None, k=5)) == 2


def test_mmr_handles_empty():
    assert mmr_select([], None, k=3) == []


# ------------------------------------------------------------------ tokenizer


def test_tokenizer_keeps_compound_drug_names_intact():
    """'piperacillin/tazobactam' is one clinical concept; splitting it turns a
    precise term into two vague ones."""
    assert "piperacillin/tazobactam" in tokenize("Give piperacillin/tazobactam IV")
    assert "beta-blocker" in tokenize("Start a beta-blocker")


def test_tokenizer_preserves_negations():
    """'not' and 'no' invert clinical meaning; a standard stoplist removes them."""
    tokens = tokenize("this is not indicated in pregnancy")
    assert "not" in tokens


# ------------------------------------------------------------------ metrics


def test_lexical_faithfulness_flags_unsupported_claims():
    contexts = ["Amoxicillin is first-line therapy for uncomplicated otitis media in children."]
    answer = "Amoxicillin is first-line for otitis media. Vancomycin cures diabetes entirely."
    result = lexical_faithfulness(answer, contexts)
    assert result.score < 1.0
    assert result.unsupported
    assert result.method == "lexical"


def test_lexical_faithfulness_rewards_supported_claims():
    contexts = ["Amoxicillin is first-line therapy for uncomplicated otitis media in children."]
    answer = "Amoxicillin is the first-line therapy for uncomplicated otitis media in children."
    assert lexical_faithfulness(answer, contexts).score == 1.0


def test_split_claims_strips_citation_markers():
    claims = split_claims("Amoxicillin is first-line therapy here [1]. Duration is five days [2].")
    assert claims
    assert not any("[1]" in c for c in claims)


def test_empty_answer_scores_zero():
    assert lexical_faithfulness("", ["ctx"]).score == 0.0
