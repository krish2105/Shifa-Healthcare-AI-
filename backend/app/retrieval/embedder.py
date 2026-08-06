"""Local embeddings with BAAI/bge-large-en-v1.5.

Two details that materially affect retrieval quality with this model family:

1. **Asymmetric prefixing.** bge expects queries to carry the instruction prefix
   "Represent this sentence for searching relevant passages: " while passages are
   embedded bare. Skipping this costs several points of recall, and it is the single
   most common way this model gets misused.

2. **L2 normalization.** We normalize at encode time so cosine similarity reduces to
   a dot product. That lets the local store use one matmul, and lets pgvector use
   the inner-product operator.
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING

import numpy as np

from app.config import settings
from app.logging_conf import get_logger

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

log = get_logger(__name__)

QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

_model: SentenceTransformer | None = None
_lock = threading.Lock()


def _resolve_device() -> str:
    if settings.embedding_device != "auto":
        return settings.embedding_device
    import torch

    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def get_model() -> SentenceTransformer:
    """Lazy, thread-safe singleton. Loading bge-large costs ~1.3 GB and a few seconds."""
    global _model
    if _model is None:
        with _lock:
            if _model is None:
                from sentence_transformers import SentenceTransformer

                device = _resolve_device()
                log.info("embedder.loading", model=settings.embedding_model, device=device)
                _model = SentenceTransformer(settings.embedding_model, device=device)
                # Half precision on accelerators only. On CPU, fp16 is emulated and
                # is slower than fp32, so enabling it there would be a pessimization.
                if settings.embedding_fp16 and device in ("mps", "cuda"):
                    _model = _model.half()
                # sentence-transformers renamed this; support both so the log line
                # does not depend on the installed minor version.
                dim_fn = getattr(
                    _model, "get_embedding_dimension", None
                ) or _model.get_sentence_embedding_dimension
                log.info(
                    "embedder.ready",
                    dim=dim_fn(),
                    device=device,
                    fp16=settings.embedding_fp16 and device in ("mps", "cuda"),
                )
    return _model


def embed_passages(texts: list[str], *, show_progress: bool = False) -> np.ndarray:
    """Embed corpus passages. No prefix — bge passages are embedded bare."""
    if not texts:
        return np.zeros((0, settings.embedding_dim), dtype=np.float32)
    vecs = get_model().encode(
        texts,
        batch_size=settings.embedding_batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=show_progress,
    )
    return np.asarray(vecs, dtype=np.float32)


def embed_query(text: str) -> np.ndarray:
    """Embed a single query, with the bge instruction prefix applied."""
    vec = get_model().encode(
        QUERY_PREFIX + text,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    return np.asarray(vec, dtype=np.float32)


def embed_queries(texts: list[str]) -> np.ndarray:
    if not texts:
        return np.zeros((0, settings.embedding_dim), dtype=np.float32)
    vecs = get_model().encode(
        [QUERY_PREFIX + t for t in texts],
        batch_size=settings.embedding_batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    return np.asarray(vecs, dtype=np.float32)
