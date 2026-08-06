"""Health, readiness, and system-info endpoints.

/health reports per-component readiness rather than a bare "ok". A stack where the
LLM is stubbed and the index is empty will happily return 200 on a naive health
check while being incapable of answering anything — so each dependency reports its
own state, and the top-level status is "degraded" whenever any of them is not real.
"""

from __future__ import annotations

import json
from typing import Any

from fastapi import APIRouter

from app.config import settings
from app.logging_conf import get_logger
from app.schemas import HealthResponse

log = get_logger("api.health")
router = APIRouter()

VERSION = "1.0.0"

DISCLAIMER = (
    "Shifa42 is a research and educational demonstration. It is not a certified "
    "medical device and does not provide medical advice. All data shown is synthetic "
    "or open, de-identified research data (MIMIC-IV-ED Demo, PhysioNet). Any real "
    "clinical use requires licensed physician oversight."
)


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    from app.llm.router import provider_info

    components: dict[str, Any] = {}

    provider = provider_info()
    components["llm"] = provider

    try:
        from app.retrieval.store import get_store

        components["vector_store"] = get_store().health()
    except Exception as exc:  # noqa: BLE001
        components["vector_store"] = {"ready": False, "error": str(exc)[:160]}

    try:
        from app.retrieval.bm25 import get_bm25

        bm = get_bm25()
        components["bm25"] = {"ready": bm is not None, "chunks": bm.count() if bm else 0}
    except Exception as exc:  # noqa: BLE001
        components["bm25"] = {"ready": False, "error": str(exc)[:160]}

    try:
        from app.retrieval.graph_store import get_graph

        kg = get_graph()
        components["knowledge_graph"] = (
            {"ready": True, **kg.stats()} if kg else {"ready": False, "reason": "not built"}
        )
    except Exception as exc:  # noqa: BLE001
        components["knowledge_graph"] = {"ready": False, "error": str(exc)[:160]}

    try:
        from app.risk.predict import model_info

        info = model_info()
        components["risk_model"] = {"ready": info["available"], "model": info.get("selected_model")}
    except Exception as exc:  # noqa: BLE001
        components["risk_model"] = {"ready": False, "error": str(exc)[:160]}

    degraded = (
        provider["degraded"]
        or not components["vector_store"].get("ready")
        or not components["bm25"].get("ready")
    )

    return HealthResponse(
        status="degraded" if degraded else "ok",
        version=VERSION,
        components=components,
        disclaimer=DISCLAIMER,
    )


@router.get("/info")
async def info() -> dict[str, Any]:
    """Corpus and model provenance — the numbers the UI displays as live stats.

    Read from the manifests written by the ingestion and training scripts, never
    hardcoded. If a number appears in the UI it was measured by a script in this
    repository.
    """
    out: dict[str, Any] = {"version": VERSION, "disclaimer": DISCLAIMER}

    for key, path in (
        ("corpus", settings.ingest_manifest_path),
        ("index", settings.processed_dir / "index_manifest.json"),
        ("ed_cohort", settings.processed_dir / "mimic_manifest.json"),
        ("risk_model", settings.risk_meta_path),
    ):
        if path.exists():
            try:
                out[key] = json.loads(path.read_text())
            except json.JSONDecodeError:
                out[key] = {"error": "manifest unreadable"}
        else:
            out[key] = None

    benchmarks = settings.artifacts_dir / "benchmark_results.json"
    out["benchmarks"] = json.loads(benchmarks.read_text()) if benchmarks.exists() else None

    out["config"] = {
        "groundedness_threshold": settings.groundedness_threshold,
        "max_reformulations": settings.max_reformulations,
        "retrieval_top_k": settings.retrieval_top_k,
        "rrf_k": settings.rrf_k,
        "mmr_lambda": settings.mmr_lambda,
        "embedding_model": settings.embedding_model,
    }
    return out


@router.get("/audit")
async def audit_log(limit: int = 50) -> dict[str, Any]:
    from app.audit.store import audit as audit_store

    limit = max(1, min(limit, 200))
    return {"entries": audit_store.recent(limit), "stats": audit_store.stats()}


@router.get("/patients")
async def patients(limit: int = 25) -> dict[str, Any]:
    from app.risk.predict import list_patients

    return {
        "patients": list_patients(max(1, min(limit, 100))),
        "note": "MIMIC-IV-ED Demo encounters — de-identified research data, no PHI.",
    }
