#!/usr/bin/env python
"""Phase A.4 — pull RxNorm ingredient names for medication normalization.

Source: RxNav REST API (U.S. National Library of Medicine). Free, no key.

Why this exists: the LLM extractor emits whatever drug name the guideline used —
brand names, abbreviations, British spellings. Without normalization,
"paracetamol", "acetaminophen" and "Tylenol" become three unconnected graph nodes,
and a query about one finds none of the others' edges. Linking medication nodes to
an RXCUI collapses them onto one identity.

We fetch the ingredient-level term list (`ttys=IN`) rather than every clinical drug
form. Ingredients are what guideline text names, and the full RxNorm release is a
large download whose branded-pack detail this project never uses.

Usage:
    python scripts/ingest_rxnorm.py
"""

from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import settings
from app.logging_conf import get_logger

log = get_logger("ingest.rxnorm")

ALLCONCEPTS = "https://rxnav.nlm.nih.gov/REST/allconcepts.json?tty=IN"


def main() -> int:
    settings.ensure_dirs()
    out_path = settings.raw_dir / "rxnorm_ingredients.json"

    if out_path.exists() and out_path.stat().st_size > 1000:
        data = json.loads(out_path.read_text())
        log.info("rxnorm.cached", ingredients=len(data), path=str(out_path))
        print(json.dumps({"cached": True, "ingredients": len(data)}, indent=2))
        return 0

    log.info("rxnorm.downloading", url=ALLCONCEPTS)
    req = urllib.request.Request(ALLCONCEPTS, headers={"User-Agent": "shifa42-research/1.0"})

    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        # A missing RxNorm link degrades graph quality; it does not break the
        # system. Fail soft and say so rather than aborting the pipeline.
        log.error("rxnorm.failed", error=str(exc)[:200],
                  effect="medication nodes will not carry RXCUIs")
        return 1

    concepts = (payload.get("minConceptGroup") or {}).get("minConcept") or []
    mapping: dict[str, str] = {}
    for c in concepts:
        name = str(c.get("name") or "").strip().lower()
        rxcui = str(c.get("rxcui") or "").strip()
        if name and rxcui:
            mapping[name] = rxcui

    out_path.write_text(json.dumps(mapping, indent=0))

    report = {
        "source": "RxNav allconcepts (tty=IN)",
        "ingredients": len(mapping),
        "path": str(out_path),
        "sample": dict(list(mapping.items())[:8]),
    }
    log.info("rxnorm.complete", ingredients=len(mapping))
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
