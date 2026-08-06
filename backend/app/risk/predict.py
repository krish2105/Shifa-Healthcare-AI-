"""Serve the ED risk model.

Lookups are by `stay_id` (or `subject_id`) against the ingested demo cohort — this
is a demonstration over a fixed open dataset, not a live feed. Every response says
so, and carries the model's own measured performance alongside the score.

Shipping the AUC and calibration slope with each prediction is deliberate. A bare
"0.71" invites more confidence than a model validated on 222 stays from 64 patients
can support, and the honest framing has to travel with the number rather than living
in a document nobody opens.
"""

from __future__ import annotations

import json
import pickle
from functools import lru_cache
from typing import Any

import pandas as pd

from app.config import settings
from app.logging_conf import get_logger
from app.risk.features import build_features, risk_band

log = get_logger("risk.predict")

MODEL_PATH = settings.artifacts_dir / "risk_model.pkl"


@lru_cache(maxsize=1)
def _load_model() -> tuple[Any, list[str]] | None:
    if not MODEL_PATH.exists():
        log.warning("risk.model_missing", path=str(MODEL_PATH),
                    hint="Run scripts/train_risk.py")
        return None
    with MODEL_PATH.open("rb") as fh:
        blob = pickle.load(fh)
    log.info("risk.model_loaded", features=len(blob["features"]))
    return blob["model"], blob["features"]


@lru_cache(maxsize=1)
def _load_meta() -> dict[str, Any]:
    if not settings.risk_meta_path.exists():
        return {}
    return json.loads(settings.risk_meta_path.read_text())


@lru_cache(maxsize=1)
def _load_cohort() -> pd.DataFrame | None:
    path = settings.processed_dir / "ed_encounters.parquet"
    if not path.exists():
        return None
    return pd.read_parquet(path)


def predict_risk(patient_id: str) -> dict[str, Any] | None:
    """Score one encounter. Returns None when the model or the patient is unavailable.

    None rather than a default: an invented risk number displayed next to real
    citations would be indistinguishable from a real one, and the composer is built
    to handle absence.
    """
    loaded = _load_model()
    cohort = _load_cohort()
    if loaded is None or cohort is None:
        return None

    model, features = loaded

    key = str(patient_id).strip()
    row = cohort[cohort["stay_id"].astype(str) == key]
    matched_on = "stay_id"
    if row.empty:
        row = cohort[cohort["subject_id"].astype(str) == key]
        matched_on = "subject_id"
        if not row.empty:
            # Several stays can share a subject; the most recent is the relevant one.
            row = row.tail(1)
    if row.empty:
        log.info("risk.patient_not_found", patient_id=key)
        return None

    X, _ = build_features(row, include_leaky=False, columns=features)

    try:
        score = float(model.predict_proba(X)[:, 1][0])
    except Exception as exc:  # noqa: BLE001
        log.error("risk.predict_failed", error=str(exc)[:200])
        return None

    meta = _load_meta()
    selected = meta.get("selected_model", "")
    perf = (meta.get("results") or {}).get(selected, {})

    r = row.iloc[0]
    return {
        "patient_id": key,
        "matched_on": matched_on,
        "risk_score": round(score, 4),
        "band": risk_band(score),
        "outcome_predicted": "admission from the emergency department",
        "model": selected,
        "top_features": list((meta.get("top_features") or {}).keys())[:5],
        "observed_triage": {
            "acuity": _num(r.get("acuity")),
            "heartrate": _num(r.get("heartrate")),
            "sbp": _num(r.get("sbp")),
            "resprate": _num(r.get("resprate")),
            "o2sat": _num(r.get("o2sat")),
            "temperature": _num(r.get("temperature")),
            "chiefcomplaint": str(r.get("chiefcomplaint") or "")[:120],
        },
        "model_performance": {
            "auc_roc": perf.get("auc_roc"),
            "auc_roc_ci95": perf.get("auc_roc_ci95"),
            "brier": perf.get("brier"),
            "calibration_slope": perf.get("calibration_slope"),
            "validation": meta.get("cv"),
            "n_stays": meta.get("n_stays"),
            "n_patients": meta.get("n_patients"),
        },
        "caveat": (
            "Demo-scale model: validated on "
            f"{meta.get('n_stays', '?')} stays from {meta.get('n_patients', '?')} patients "
            "in MIMIC-IV-ED Demo. Directional only — not fit for clinical use."
        ),
    }


def _num(v: object) -> float | None:
    try:
        f = float(v)  # type: ignore[arg-type]
        return None if pd.isna(f) else round(f, 2)
    except (TypeError, ValueError):
        return None


def list_patients(limit: int = 25) -> list[dict[str, Any]]:
    """Sample encounters, so the UI has real ids to demo against."""
    cohort = _load_cohort()
    if cohort is None:
        return []
    sample = cohort.head(limit)
    return [
        {
            "stay_id": str(r.stay_id),
            "subject_id": str(r.subject_id),
            "acuity": _num(r.acuity),
            "chiefcomplaint": str(getattr(r, "chiefcomplaint", "") or "")[:80],
            "disposition": str(r.disposition),
        }
        for r in sample.itertuples(index=False)
    ]


def model_info() -> dict[str, Any]:
    meta = _load_meta()
    return {
        "available": MODEL_PATH.exists(),
        "selected_model": meta.get("selected_model"),
        "results": meta.get("results", {}),
        "baselines": meta.get("baselines", {}),
        "n_stays": meta.get("n_stays"),
        "n_patients": meta.get("n_patients"),
        "cv": meta.get("cv"),
        "selection_rule": meta.get("selection_rule"),
        "leakage_check": meta.get("leakage_check"),
    }
