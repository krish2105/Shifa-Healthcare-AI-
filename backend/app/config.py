"""Central configuration.

Every tunable in Shifa42 lives here so that no number in the pipeline is a magic
constant buried in a function. The critic threshold in particular is a governance
parameter, not an implementation detail — RESULTS.md reports how it trades off
against the escalation rate, which is only honest if it is a single named knob.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = BACKEND_ROOT / "data"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(BACKEND_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ---------------------------------------------------------------- app
    app_env: Literal["local", "ci", "production"] = "local"
    log_level: str = "INFO"
    api_prefix: str = "/api"

    cors_origins: list[str] = Field(
        default=["http://localhost:3000", "http://127.0.0.1:3000"],
        description="Locked allow-list. Never '*' — this API streams clinical content.",
    )
    rate_limit: str = Field(
        default="30/minute",
        description="slowapi limit applied to the query endpoint.",
    )

    # ---------------------------------------------------------------- LLM
    # Resolution order is groq -> gemini -> cerebras -> stub. See app/llm/router.py.
    groq_api_key: str | None = None
    gemini_api_key: str | None = None
    cerebras_api_key: str | None = None

    groq_model: str = "groq/llama-3.3-70b-versatile"
    groq_model_fast: str = "groq/llama-3.1-8b-instant"
    gemini_model: str = "gemini/gemini-2.0-flash"
    cerebras_model: str = "cerebras/llama3.1-70b"

    llm_timeout_s: float = 60.0
    llm_max_retries: int = 3
    llm_temperature: float = 0.1

    # ---------------------------------------------------------------- storage
    database_url: str | None = Field(
        default=None,
        description="Neon/Postgres URL. When unset the LocalVectorStore is used.",
    )
    db_pool_min: int = 1
    db_pool_max: int = 5

    # ---------------------------------------------------------------- retrieval
    embedding_model: str = "BAAI/bge-large-en-v1.5"
    embedding_dim: int = 1024
    embedding_device: Literal["auto", "cpu", "mps", "cuda"] = "auto"
    # Measured on an M4 Pro over this corpus (mean ~460 tokens/chunk):
    #   fp32 bs=32  ->  7.6 chunks/s
    #   fp16 bs=128 ->  9.3 chunks/s
    # fp16 is a ~22% speedup for no meaningful retrieval cost — embeddings are
    # L2-normalized immediately afterwards, so half-precision rounding lands far
    # below the granularity that affects ranking. Larger batches past 128 stop
    # helping; the MPS backend is memory-bandwidth bound here, not compute bound.
    embedding_batch_size: int = 128
    embedding_fp16: bool = True

    child_chunk_tokens: int = 450
    child_chunk_overlap: int = 60
    parent_chunk_tokens: int = 2000

    retrieval_top_k: int = 12
    retrieval_candidate_k: int = 40
    rrf_k: int = 60  # Reciprocal Rank Fusion damping constant
    mmr_lambda: float = 0.6  # 1.0 = pure relevance, 0.0 = pure diversity
    graph_max_hops: int = 2
    graph_max_nodes: int = 25

    # ---------------------------------------------------------------- critic
    groundedness_threshold: float = Field(
        default=0.75,
        ge=0.0,
        le=1.0,
        description="Below this, the draft is reformulated and retried, then escalated.",
    )
    max_reformulations: int = Field(default=2, ge=0, le=5)

    # ---------------------------------------------------------------- corpus
    guideline_doc_limit: int = 2000
    min_guideline_chars: int = 1200

    @field_validator("cors_origins", mode="before")
    @classmethod
    def _split_origins(cls, v: object) -> object:
        if isinstance(v, str):
            return [o.strip() for o in v.split(",") if o.strip()]
        return v

    # ---------------------------------------------------------------- paths
    @property
    def raw_dir(self) -> Path:
        return DATA_DIR / "raw"

    @property
    def processed_dir(self) -> Path:
        return DATA_DIR / "processed"

    @property
    def artifacts_dir(self) -> Path:
        return DATA_DIR / "artifacts"

    @property
    def chunks_path(self) -> Path:
        return self.processed_dir / "chunks.parquet"

    @property
    def parents_path(self) -> Path:
        return self.processed_dir / "parents.parquet"

    @property
    def graph_path(self) -> Path:
        return self.artifacts_dir / "knowledge_graph.pkl"

    @property
    def bm25_path(self) -> Path:
        return self.artifacts_dir / "bm25.pkl"

    @property
    def local_vectors_path(self) -> Path:
        return self.artifacts_dir / "vectors.npz"

    @property
    def risk_model_path(self) -> Path:
        return self.artifacts_dir / "risk_model.json"

    @property
    def risk_meta_path(self) -> Path:
        return self.artifacts_dir / "risk_meta.json"

    @property
    def ingest_manifest_path(self) -> Path:
        """Measured counts from the last ingest. Nothing is reported that isn't here."""
        return self.processed_dir / "ingest_manifest.json"

    def ensure_dirs(self) -> None:
        for p in (self.raw_dir, self.processed_dir, self.artifacts_dir):
            p.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
