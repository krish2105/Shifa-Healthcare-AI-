"""Feature engineering for ED risk stratification.

Shared by `scripts/train_risk.py` and `app/risk/predict.py` on purpose. Training/
serving skew — where the serving path rebuilds features slightly differently from
the training path — is one of the most common and least visible ways an ML system
degrades in production. One function, both callers.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

TARGET = "admitted"

TRIAGE_NUMERIC = ["temperature", "heartrate", "resprate", "o2sat", "sbp", "dbp", "pain", "acuity"]
TRIAGE_CATEGORICAL = ["gender", "arrival_transport"]

# Measured during the stay, i.e. after the admission decision. Excluded from the
# model; see scripts/train_risk.py for the leakage discussion.
LEAKY = ["los_hours", "n_vital_checks", "hr_std", "sbp_min", "o2sat_min", "n_diagnoses"]

# A bag-of-words model over 222 rows would memorize the sample. A short list of
# high-acuity presentations is the amount of signal this dataset can support.
COMPLAINT_FLAGS = {
    "cc_chest_pain": ("chest pain", "chest pressure"),
    "cc_dyspnea": ("dyspnea", "shortness of breath", "sob", "breathing"),
    "cc_abdominal": ("abd pain", "abdominal pain"),
    "cc_neuro": ("weakness", "altered", "syncope", "seizure", "stroke", "dizziness"),
    "cc_fever": ("fever", "sepsis", "infection"),
    "cc_trauma": ("fall", "trauma", "injury", "fracture", "mvc"),
    "cc_cardiac": ("palpitation", "afib", "cardiac", "hypotension"),
}

_DIGITS = re.compile(r"\d+")


def coerce_pain(v: object) -> float:
    """Pain is free text: '7', '7/10', 'denies', 'unable'. Extract a 0-10 number."""
    if pd.isna(v):
        return np.nan
    s = str(v).strip().lower()
    if s in ("denies", "none", "no", "0"):
        return 0.0
    m = _DIGITS.search(s)
    return min(10.0, float(m.group(0))) if m else np.nan


def build_features(
    df: pd.DataFrame,
    include_leaky: bool = False,
    columns: list[str] | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Build the model matrix.

    `columns` pins the output to a known feature list, which serving must do: a
    single-row request cannot regenerate the same one-hot columns that the full
    training frame produced, so missing ones are filled with zeros and unexpected
    ones dropped.
    """
    X = pd.DataFrame(index=df.index)

    for col in TRIAGE_NUMERIC:
        if col == "pain":
            X[col] = df[col].map(coerce_pain) if col in df.columns else np.nan
        else:
            X[col] = pd.to_numeric(df[col], errors="coerce") if col in df.columns else np.nan

    # Shock index (HR/SBP) and pulse pressure carry instability neither component
    # shows alone — a patient can have a normal heart rate and a normal blood
    # pressure while their ratio is alarming.
    with np.errstate(divide="ignore", invalid="ignore"):
        X["shock_index"] = X["heartrate"] / X["sbp"].replace(0, np.nan)
        X["pulse_pressure"] = X["sbp"] - X["dbp"]

    for col in TRIAGE_CATEGORICAL:
        if col in df.columns:
            dummies = pd.get_dummies(df[col].astype(str), prefix=col, dtype=float)
            for c in dummies.columns:
                # Rare categories are noise at this sample size. When `columns` is
                # pinned we keep whatever the trained model expects instead.
                if columns is not None or dummies[c].sum() >= 10:
                    X[c] = dummies[c]

    cc = df.get("chiefcomplaint", pd.Series("", index=df.index)).astype(str).str.lower()
    for flag, keywords in COMPLAINT_FLAGS.items():
        X[flag] = cc.apply(lambda s, kw=keywords: float(any(k in s for k in kw)))

    if include_leaky:
        for col in LEAKY:
            if col in df.columns:
                X[f"LEAKY_{col}"] = pd.to_numeric(df[col], errors="coerce")

    if columns is not None:
        for c in columns:
            if c not in X.columns:
                X[c] = 0.0
        X = X[columns]

    return X, list(X.columns)


def risk_band(score: float) -> str:
    """Map a probability to a display band.

    Thresholds are display conventions, not validated clinical cut-points, and are
    labelled as such wherever they surface.
    """
    if score >= 0.75:
        return "high"
    if score >= 0.5:
        return "elevated"
    if score >= 0.25:
        return "moderate"
    return "low"
