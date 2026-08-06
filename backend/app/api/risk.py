"""GET /risk-score/{patient_id} — ED risk stratification."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException

from app.logging_conf import get_logger
from app.schemas import RiskResponse

log = get_logger("api.risk")
router = APIRouter()


@router.get("/risk-score/{patient_id}", response_model=RiskResponse)
async def risk_score(patient_id: str) -> Any:
    if not patient_id.isalnum() or len(patient_id) > 64:
        raise HTTPException(status_code=422, detail="patient_id must be alphanumeric, <= 64 chars")

    from app.risk.predict import model_info, predict_risk

    result = predict_risk(patient_id)
    if result is None:
        info = model_info()
        if not info["available"]:
            raise HTTPException(
                status_code=503,
                detail="Risk model not trained. Run scripts/train_risk.py.",
            )
        raise HTTPException(
            status_code=404,
            detail=f"No encounter found for '{patient_id}' in the MIMIC-IV-ED Demo cohort. "
            "See GET /api/patients for valid ids.",
        )
    return result


@router.get("/risk-model")
async def risk_model() -> dict[str, Any]:
    """Full model card: metrics, baselines, validation scheme, leakage check."""
    from app.risk.predict import model_info

    return model_info()
