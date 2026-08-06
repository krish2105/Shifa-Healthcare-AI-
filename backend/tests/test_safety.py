"""Safety-layer tests: identifier screening, critic gating, and the no-PHI claim.

These are the tests that matter most in this project. The retrieval tests catch
quality regressions; these catch the failures that would make the project's central
claims false.
"""

from __future__ import annotations

import json

import pytest

from app.agent.nodes.critic import critic_decision
from app.agent.nodes.intake import screen_identifiers
from app.config import settings

# ------------------------------------------------------------- PHI screening


@pytest.mark.parametrize(
    "text,expected",
    [
        ("Patient SSN 123-45-6789 presents with chest pain", "ssn"),
        ("MRN: A9928471 admitted overnight", "mrn"),
        ("Contact at 555-867-5309 for follow up", "phone"),
        ("Email jane.doe@hospital.org about results", "email"),
        ("DOB: 1974-03-11, presenting with dyspnea", "dob"),
    ],
)
def test_identifier_patterns_are_detected(text: str, expected: str):
    assert expected in screen_identifiers(text)


@pytest.mark.parametrize(
    "text",
    [
        "What is the first-line antibiotic for community-acquired pneumonia?",
        "Give 500 mg amoxicillin three times daily for 5 days",
        "A 67-year-old presents with a heart rate of 112 and BP 88/54",
        "Target MAP 65 mmHg within 1 hour per the 2021 guideline",
    ],
)
def test_ordinary_clinical_text_does_not_trigger_the_flag(text: str):
    """A screen that fires on doses, ages and vitals trains readers to ignore it."""
    assert screen_identifiers(text) == []


# ----------------------------------------------------------- critic gating


def test_passing_score_routes_to_composer():
    state = {"groundedness": 0.9, "attempt": 1}
    assert critic_decision(state) == "final_composer"


def test_score_exactly_at_threshold_passes():
    """The threshold is inclusive; an off-by-one here silently raises the bar."""
    state = {"groundedness": settings.groundedness_threshold, "attempt": 1}
    assert critic_decision(state) == "final_composer"


def test_failing_score_with_retries_left_reformulates():
    state = {"groundedness": 0.3, "attempt": 1}
    assert critic_decision(state) == "reformulate"


def test_failing_score_with_retries_exhausted_escalates():
    state = {"groundedness": 0.3, "attempt": settings.max_reformulations + 1}
    assert critic_decision(state) == "escalate"


def test_missing_score_fails_closed():
    """An absent score must escalate, never pass. Verification that fails open is not
    verification."""
    assert critic_decision({"attempt": 99}) == "escalate"


def test_retry_budget_is_finite():
    """Guards against an infinite reformulate loop if the cap is ever mis-set."""
    decisions = [
        critic_decision({"groundedness": 0.0, "attempt": a})
        for a in range(1, settings.max_reformulations + 6)
    ]
    assert "escalate" in decisions


# ------------------------------------------------------------- no PHI on disk

_FORBIDDEN_SUBSTRINGS = ("name", "address", "phone", "email", "mrn", "ssn", "dob")


def test_ed_cohort_carries_no_identifier_columns():
    """Enforces the README's no-PHI claim against the data actually on disk."""
    path = settings.processed_dir / "ed_encounters.parquet"
    if not path.exists():
        pytest.skip("ED cohort not ingested; run scripts/ingest_mimic.py")

    import pandas as pd

    df = pd.read_parquet(path)
    offending = [c for c in df.columns if any(f in c.lower() for f in _FORBIDDEN_SUBSTRINGS)]
    assert not offending, f"identifier-shaped columns present: {offending}"


def test_ed_cohort_manifest_reports_open_licence():
    path = settings.processed_dir / "mimic_manifest.json"
    if not path.exists():
        pytest.skip("ED cohort not ingested")
    manifest = json.loads(path.read_text())
    assert "ODbL" in manifest.get("license", "")


def test_risk_model_excludes_race_as_a_predictor():
    """Race is present in the source data. Using it would encode access disparities
    as clinical facts."""
    if not settings.risk_meta_path.exists():
        pytest.skip("Risk model not trained; run scripts/train_risk.py")
    meta = json.loads(settings.risk_meta_path.read_text())
    assert meta.get("race_used_as_predictor") is False
    assert not any("race" in f.lower() for f in meta.get("features", []))


def test_risk_model_excludes_post_triage_leakage():
    if not settings.risk_meta_path.exists():
        pytest.skip("Risk model not trained")
    meta = json.loads(settings.risk_meta_path.read_text())
    if meta.get("leaky_features_included"):
        pytest.skip("run was explicitly --include-leaky")
    assert not any(f.startswith("LEAKY_") for f in meta.get("features", []))


def test_risk_model_validated_with_patient_grouping():
    """222 stays come from 64 patients — an ungrouped split leaks."""
    if not settings.risk_meta_path.exists():
        pytest.skip("Risk model not trained")
    meta = json.loads(settings.risk_meta_path.read_text())
    assert "GroupKFold" in meta.get("cv", "")
