#!/usr/bin/env python
"""Phase A.2 — download the MIMIC-IV-ED Demo and assemble the ED encounter table.

Source: PhysioNet MIMIC-IV-ED Demo v2.2. Real, de-identified emergency-department
records from Beth Israel Deaconess. Open access under ODbL — no credentialing, no
data use agreement, which is precisely why this subset was chosen over full MIMIC-IV.

**No PHI.** The demo is already de-identified at source: dates are shifted, ages over
89 are aggregated, and no names, MRNs or addresses exist in the files. This script
additionally drops every free-text field except `chiefcomplaint`, and asserts that
the assembled table carries no identifier-shaped columns before writing it.

Usage:
    python scripts/ingest_mimic.py
"""

from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from app.config import settings
from app.logging_conf import get_logger

log = get_logger("ingest.mimic")

BASE = "https://physionet.org/files/mimic-iv-ed-demo/2.2/ed"
FILES = ["edstays", "triage", "diagnosis", "vitalsign", "medrecon", "pyxis"]

# Anything matching these names must never reach the processed table.
_FORBIDDEN_COLS = {"name", "firstname", "lastname", "address", "phone", "email", "mrn", "ssn", "dob"}


def download() -> dict[str, Path]:
    out_dir = settings.raw_dir / "mimic_iv_ed_demo"
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    for name in FILES:
        dest = out_dir / f"{name}.csv.gz"
        if dest.exists() and dest.stat().st_size > 0:
            log.info("mimic.cached", file=name, bytes=dest.stat().st_size)
        else:
            url = f"{BASE}/{name}.csv.gz"
            log.info("mimic.downloading", file=name, url=url)
            req = urllib.request.Request(url, headers={"User-Agent": "shifa42-research/1.0"})
            with urllib.request.urlopen(req, timeout=120) as resp, dest.open("wb") as fh:
                fh.write(resp.read())
            log.info("mimic.downloaded", file=name, bytes=dest.stat().st_size)
        paths[name] = dest
    return paths


def assert_no_phi(df: pd.DataFrame) -> None:
    bad = [c for c in df.columns if any(f in c.lower() for f in _FORBIDDEN_COLS)]
    if bad:
        raise AssertionError(f"Refusing to write: identifier-shaped columns present: {bad}")


def main() -> int:
    settings.ensure_dirs()
    paths = download()

    edstays = pd.read_csv(paths["edstays"], compression="gzip")
    triage = pd.read_csv(paths["triage"], compression="gzip")
    diagnosis = pd.read_csv(paths["diagnosis"], compression="gzip")
    vitals = pd.read_csv(paths["vitalsign"], compression="gzip")

    log.info(
        "mimic.loaded",
        edstays=len(edstays), triage=len(triage),
        diagnosis=len(diagnosis), vitalsign=len(vitals),
        patients=edstays["subject_id"].nunique(),
    )

    # -- join triage onto stays ------------------------------------------------
    df = edstays.merge(triage, on=["subject_id", "stay_id"], how="left", suffixes=("", "_triage"))

    # -- length of stay in hours ----------------------------------------------
    for col in ("intime", "outtime"):
        df[col] = pd.to_datetime(df[col], errors="coerce")
    df["los_hours"] = (df["outtime"] - df["intime"]).dt.total_seconds() / 3600.0

    # -- diagnosis count per stay ---------------------------------------------
    dx_counts = diagnosis.groupby("stay_id").size().rename("n_diagnoses")
    df = df.merge(dx_counts, on="stay_id", how="left")
    df["n_diagnoses"] = df["n_diagnoses"].fillna(0).astype(int)

    # -- vitals volatility -----------------------------------------------------
    # A patient whose vitals are re-measured many times, or whose heart rate swings,
    # is being watched closely. That is signal about clinical concern which the
    # single triage snapshot does not carry.
    if len(vitals):
        v = vitals.groupby("stay_id").agg(
            n_vital_checks=("stay_id", "size"),
            hr_std=("heartrate", "std"),
            sbp_min=("sbp", "min"),
            o2sat_min=("o2sat", "min"),
        )
        df = df.merge(v, on="stay_id", how="left")

    # -- target ----------------------------------------------------------------
    # Admission, not the triage acuity score. Acuity is assigned by the triage nurse
    # at the same moment the vitals are recorded, so predicting it from those vitals
    # is close to predicting a human's summary of the very same inputs — high AUC,
    # near-zero clinical meaning. Admission is a downstream outcome decided later by
    # different people, which makes it a real prediction rather than a restatement.
    df["admitted"] = (df["disposition"].astype(str).str.upper() == "ADMITTED").astype(int)

    keep = [
        "subject_id", "stay_id", "gender", "race", "arrival_transport", "disposition",
        "temperature", "heartrate", "resprate", "o2sat", "sbp", "dbp", "pain", "acuity",
        "chiefcomplaint", "los_hours", "n_diagnoses", "n_vital_checks",
        "hr_std", "sbp_min", "o2sat_min", "admitted",
    ]
    df = df[[c for c in keep if c in df.columns]].copy()

    assert_no_phi(df)

    out = settings.processed_dir / "ed_encounters.parquet"
    df.to_parquet(out, index=False)

    manifest = {
        "dataset": "MIMIC-IV-ED Demo v2.2 (PhysioNet)",
        "license": "ODbL — open access, no credentialing",
        "stays": len(df),
        "patients": int(df["subject_id"].nunique()),
        "admitted": int(df["admitted"].sum()),
        "admitted_rate": round(float(df["admitted"].mean()), 4),
        "disposition_counts": df["disposition"].value_counts().to_dict(),
        "acuity_counts": (
            df["acuity"].value_counts(dropna=False).astype(int).to_dict() if "acuity" in df else {}
        ),
        "missingness": {c: round(float(df[c].isna().mean()), 4) for c in df.columns},
        "columns": list(df.columns),
    }
    (settings.processed_dir / "mimic_manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str)
    )

    log.info("mimic.complete", stays=len(df), patients=df["subject_id"].nunique())
    print("\n" + json.dumps(manifest, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
